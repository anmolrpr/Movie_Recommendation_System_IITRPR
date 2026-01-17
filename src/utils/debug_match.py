import asyncio
from src.services import RecommenderService
from src.models import User
from src.utils.llm_helper import LLMService
import numpy as np

async def test():
    print("--- 1. Testing LLMService ---")
    llm = LLMService()
    mood = "horror"
    vec = llm.get_mood_features(mood)
    print(f"Mood: '{mood}'")
    print(f"Vector sum: {vec.sum()}")
    print(f"Active indices: {np.where(vec > 0)[0]}")
    
    genres = llm.genres
    active_genres = [genres[i] for i in np.where(vec > 0)[0]]
    print(f"Mapped Genres: {active_genres}")
    
    if "Horror" not in active_genres:
        print("ERROR: 'Horror' not found in mapped genres!")
    else:
        print("SUCCESS: 'Horror' mapped correctly.")

    print("\n--- 2. Testing Recommender Service Logic ---")
    rec = RecommenderService()
    rec.initialize()
    
    # Fake user
    user = User(username="test_debug", password_hash="abc")
    user.id = "fake_id"
    
    # We need to mock interaction finding or just rely on the fact that for new user it's empty
    # But get_recommendation queries DB. We can't easily mock DB here without setup.
    # So we'll trust the unit test or run it against the real DB if possible.
    
    # Let's just check if ANY horror movies exist in the dataset
    horror_idx = genres.index("Horror")
    
    count_horror = 0
    for _, row in rec.data_loader.items.iterrows():
        if row['Horror'] == 1:
            count_horror += 1
            
    print(f"Total Horror movies in dataset: {count_horror}")
    
    if count_horror == 0:
        print("CRITICAL: No horror movies in dataset!")

if __name__ == "__main__":
    asyncio.run(test())
