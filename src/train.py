import numpy as np
import torch
import matplotlib.pyplot as plt
from src.env.recsys_env import RecSysEnv
from src.agent.dqn import DQNAgent
import os

def train():
    env = RecSysEnv(history_len=5)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim, lr=1e-4, gamma=0.95)
    
    episodes = 500
    batch_size = 64
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995
    epsilon = epsilon_start
    
    scores = []
    
    print(f"Starting training on {env.data_loader.get_stats()}...")
    
    for e in range(episodes):
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        scores.append(score)
        
        if e % 10 == 0:
            print(f"Episode {e}/{episodes} | Score: {score:.2f} | Epsilon: {epsilon:.2f} | Memory: {len(agent.memory)}")
            
    # Save Model
    if not os.path.exists("models"):
        os.makedirs("models")
    agent.save("models/dqn_rec.pth")
    print("Model saved to models/dqn_rec.pth")
    
    # Plot
    plt.plot(scores)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("models/training_curve.png")
    print("Training curve saved.")

if __name__ == "__main__":
    train()
