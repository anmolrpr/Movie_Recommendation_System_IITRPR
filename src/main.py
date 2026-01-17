from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from src.db import init_db
from src.models import User, Interaction
from src.auth import get_password_hash, verify_password, create_access_token, decode_access_token
from contextlib import asynccontextmanager
from typing import Annotated
from dotenv import load_dotenv
import os

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    recommender.initialize()
    yield
    # Shutdown

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(lifespan=lifespan)

# Mount Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    payload = decode_access_token(token)
    if not payload:
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    username: str = payload.get("sub")
    user = await User.find_one(User.username == username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

@app.get("/")
async def root():
    return FileResponse('static/index.html')

# --- Auth Routes ---

@app.post("/register")
async def register(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    existing_user = await User.find_one(User.username == form_data.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_pw = get_password_hash(form_data.password)
    new_user = User(username=form_data.username, password_hash=hashed_pw)
    await new_user.insert()
    
    return {"message": "User created successfully"}

@app.post("/token")
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = await User.find_one(User.username == form_data.username)
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: Annotated[User, Depends(get_current_user)]):
    return {"username": current_user.username, "id": str(current_user.id)}

# --- Recommendation Routes ---
from src.services import recommender
from pydantic import BaseModel

class RecommendRequest(BaseModel):
    mood: str

class RateRequest(BaseModel):
    movie_id: int
    rating: int

@app.post("/recommend")
async def get_recommendation(
    request: RecommendRequest, 
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Get 5 movie recommendations based on User History + Mood.
    """
    # Returns List[dict]
    results = await recommender.get_recommendation(current_user, request.mood)
    return results

@app.post("/rate")
async def rate_movie(
    request: RateRequest,
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Submit a rating for a movie.
    """
    # Create Interaction
    interaction = Interaction(
        user_id=str(current_user.id),
        movie_id=request.movie_id,
        rating=request.rating
    )
    await interaction.insert()
    return {"message": "Rating saved"}
