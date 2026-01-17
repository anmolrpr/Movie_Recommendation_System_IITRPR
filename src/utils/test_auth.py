import requests
import time

BASE_URL = "http://localhost:8000"

def test_auth():
    # Wait for server
    time.sleep(2)
    
    # 1. Register
    username = f"user_{int(time.time())}"
    password = "password123"
    
    print(f"Registering {username}...")
    resp = requests.post(f"{BASE_URL}/register", data={"username": username, "password": password})
    if resp.status_code == 200:
        print("Registration Success:", resp.json())
    else:
        print("Registration Failed:", resp.text)
        return

    # 2. Login
    print("Logging in...")
    resp = requests.post(f"{BASE_URL}/token", data={"username": username, "password": password})
    if resp.status_code == 200:
        token_data = resp.json()
        print("Login Success. Token received.")
        access_token = token_data["access_token"]
    else:
        print("Login Failed:", resp.text)
        return

    # 3. Protected Route
    print("Accessing protected route...")
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.get(f"{BASE_URL}/users/me", headers=headers)
    if resp.status_code == 200:
        print("Protected verification passed:", resp.json())
    else:
        print("Protected Access Failed:", resp.text)

if __name__ == "__main__":
    test_auth()
