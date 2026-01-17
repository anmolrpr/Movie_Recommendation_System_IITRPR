import asyncio
from src.db import init_db
from src.models import User

async def check():
    try:
        print("Attempting to connect to MongoDB...")
        await init_db()
        print("Connection Successful!")
        
        # Try to find a user just to check read access
        try:
            u = await User.find_one(User.username == "test")
            print("Read check passed.")
        except Exception as e:
            print(f"Read check warning: {e}")
            
    except Exception as e:
        print(f"Connection Failed: {e}")
        print("Please ensure MongoDB is running or MONGO_URI is set.")

if __name__ == "__main__":
    asyncio.run(check())
