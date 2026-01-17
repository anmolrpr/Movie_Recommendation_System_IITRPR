from src.env.recsys_env import RecSysEnv
import numpy as np

def test_env():
    env = RecSysEnv()
    state, info = env.reset()
    
    print("State Shape:", state.shape)
    print("Action Space:", env.action_space.n)
    print("Initial State Example (First 10):", state[:10])
    
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action = env.action_space.sample() # Random action
        next_state, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if steps % 2 == 0:
            print(f"Step {steps}: Action {action}, Reward {reward}")
            
        done = terminated or truncated
        
    print(f"Episode finished after {steps} steps. Total Reward: {total_reward}")

if __name__ == "__main__":
    test_env()
