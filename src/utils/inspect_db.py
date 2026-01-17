import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os

DATABASE_URL = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = "recsys_db"

async def inspect():
    client = AsyncIOMotorClient(DATABASE_URL)
    db = client[DATABASE_NAME]
    
    print(f"--- Inspecting DB: {DATABASE_NAME} ---")
    
    collections = await db.list_collection_names()
    print(f"Collections found: {collections}")
    
    for col_name in collections:
        count = await db[col_name].count_documents({})
        print(f"Collection '{col_name}': {count} documents")
        
        if count > 0:
            # Show sample
            sample = await db[col_name].find_one()
            print(f"  Sample: {sample}")

    if not collections:
        print("WARNING: No collections found in this database!")

if __name__ == "__main__":
    asyncio.run(inspect())
