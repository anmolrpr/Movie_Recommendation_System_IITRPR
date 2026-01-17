from beanie import Document
from pydantic import Field
from datetime import datetime
from typing import Optional

class User(Document):
    username: str = Field(unique=True)
    password_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "users"

class Interaction(Document):
    user_id: str # References User.id
    movie_id: int
    rating: int # 1-5
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "interactions"
