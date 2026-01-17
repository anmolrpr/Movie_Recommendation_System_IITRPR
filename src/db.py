import motor.motor_asyncio
from beanie import init_beanie
from src.models import User, Interaction
import os

# Allow overriding DB URL for different environments
DATABASE_URL = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = "recsys_db"

async def init_db():
    client = motor.motor_asyncio.AsyncIOMotorClient(DATABASE_URL)
    await init_beanie(database=client[DATABASE_NAME], document_models=[User, Interaction])
    print(f"Connected to MongoDB at {DATABASE_URL} ({DATABASE_NAME})")
