import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_full_loop():
    time.sleep(2)
    # 1. Register/Login
    username = f"user_{int(time.time())}"
    password = "password123"
    
    print(f"Creating user {username}...")
    requests.post(f"{BASE_URL}/register", data={"username": username, "password": password})
    
    token_resp = requests.post(f"{BASE_URL}/token", data={"username": username, "password": password})
    access_token = token_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # 2. Get Recommendation (Cold Start - No History)
    print("\n[Request 1] Mood: 'I want a funny movie'")
    rec_resp = requests.post(f"{BASE_URL}/recommend", json={"mood": "I want a funny movie"}, headers=headers)
    print("Recommendation:", json.dumps(rec_resp.json(), indent=2))
    rec_data = rec_resp.json()
    movie_id = rec_data.get("movie_id")
    
    if not movie_id:
        print("Error: No movie returned")
        return

    # 3. Rate It (e.g., 5 stars)
    print(f"\n[Rate] Giving 5 stars to Movie ID {movie_id}...")
    rate_resp = requests.post(f"{BASE_URL}/rate", json={"movie_id": movie_id, "rating": 5}, headers=headers)
    print("Response:", rate_resp.json())
    
    # 4. Get Next Recommendation (With History)
    print("\n[Request 2] Mood: 'Something scary now'")
    rec_resp_2 = requests.post(f"{BASE_URL}/recommend", json={"mood": "Something scary now"}, headers=headers)
    print("Recommendation:", json.dumps(rec_resp_2.json(), indent=2))

if __name__ == "__main__":
    test_full_loop()
