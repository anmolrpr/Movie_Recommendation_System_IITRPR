import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from src.data.movielens import MovieLens1M

class RecSysEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path="data/ml-1m", history_len=5):
        super(RecSysEnv, self).__init__()
        
        self.data_loader = MovieLens1M(data_dir=data_path)
        self.history_len = history_len
        
        # Action Space: Recommend one of the movies (Discrete)
        # Note: Movie IDs are not 0-indexed contiguous, so we map them
        self.movie_ids = self.data_loader.get_all_movie_ids()
        self.movie_id_to_index = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.index_to_movie_id = {i: mid for mid, i in self.movie_id_to_index.items()}
        self.num_movies = len(self.movie_ids)
        
        self.action_space = spaces.Discrete(self.num_movies)
        
        # State Space: 
        # History (N movies * 18 genres) + Context (18 genres)
        # We use flattened vector for generic compat
        self.genre_dim = 18 # Updated for ML-1M
        state_dim = (self.history_len * self.genre_dim) + self.genre_dim
        self.observation_space = spaces.Box(low=0, high=1, shape=(state_dim,), dtype=np.float32)
        
        self.current_user = None
        self.user_history = [] # List of movie_ids
        self.target_context = None # Genre vector of "Mood"
        self.steps_taken = 0
        self.max_steps = 10 # Session length

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Pick a random user
        self.current_user = np.random.choice(self.data_loader.ratings['user_id'].unique())
        
        # 2. Get their full history
        full_history = self.data_loader.get_user_history(self.current_user)
        
        # 3. Simulate "History" vs "Future"
        # We need at least history_len + 1 items
        if len(full_history) < self.history_len + 5:
            return self.reset(seed=seed) # Retry if user has too few ratings
            
        # Split point
        split_idx = np.random.randint(self.history_len, len(full_history) - 2)
        
        # Historical interactions
        self.past_interactions = full_history.iloc[:split_idx]
        self.future_interactions = full_history.iloc[split_idx:]
        
        # Initial State History (Last N from past)
        # We take the movie IDs
        self.state_history = self.past_interactions.tail(self.history_len)['movie_id'].values.tolist()
        
        # 4. Determine Context (Goal)
        # We pick a "Target Movie" from the future to set the mood
        target_row = self.future_interactions.iloc[0]
        self.target_movie_id = target_row['movie_id']
        self.target_context = self.data_loader.get_movie_features(self.target_movie_id)
        
        self.steps_taken = 0
        
        return self._get_state(), {}

    def _get_state(self):
        # Flatten history genres
        history_vecs = []
        for mid in self.state_history:
            vec = self.data_loader.get_movie_features(mid)
            history_vecs.append(vec)
        
        # If history is shorter than N (shouldn't happen with logic above but safe-guard)
        while len(history_vecs) < self.history_len:
            history_vecs.insert(0, np.zeros(self.genre_dim))
            
        history_flat = np.concatenate(history_vecs)
        
        # Append Context
        state = np.concatenate([history_flat, self.target_context])
        return state.astype(np.float32)

    def step(self, action_idx):
        self.steps_taken += 1
        movie_id = self.index_to_movie_id[action_idx]
        
        # Calculate Reward
        # 1. Check if it matches the "Mood" (Context Similarity)
        movie_features = self.data_loader.get_movie_features(movie_id)
        # Cosine similarity roughly (or dot product since binary)
        similarity = np.dot(movie_features, self.target_context) / (np.linalg.norm(movie_features) * np.linalg.norm(self.target_context) + 1e-9)
        
        # 2. Check if user actually liked it (Ground Truth from Future)
        # Use Rating if available, else 0
        user_rating_row = self.future_interactions[self.future_interactions['movie_id'] == movie_id]
        
        base_reward = 0
        if not user_rating_row.empty:
            actual_rating = user_rating_row.iloc[0]['rating']
            base_reward = actual_rating # 1 to 5
        else:
            # Penalty for recommending irrelevant/unseen movie
            base_reward = 0 
            
        # Composite Reward
        # Weighted: 70% User Rating, 30% Mood Match
        # If user didn't rate it, we rely heavily on Mood Match?
        # Let's keep it simple: Reward = ActualRating if exists, else Similarity * 3
        
        final_reward = 0
        if not user_rating_row.empty:
            final_reward = actual_rating
        else:
            # New Hybrid Reward: Mood Match + Quality (Avg Rating)
            # Normalize Avg Rating (1-5) to roughly 0-1 scale or keep as bonus
            avg_rating = self.data_loader.get_average_rating(movie_id)
            
            # Formula: 50% Mood Match (max 1.0) + 50% Quality (max 5.0 scaled down)
            # Let's say we want max reward to be around 3.0 for unseen movies
            # Similarity is 0-1. AvgRating is 1-5.
            
            # (Similarity * 1.5) + (AvgRating * 0.3)
            # If Sim=1.0, Avg=5.0 -> 1.5 + 1.5 = 3.0
            # If Sim=0.0, Avg=5.0 -> 0 + 1.5 = 1.5 (Still rewarded for quality)
            # If Sim=0.0, Avg=1.0 -> 0 + 0.3 = 0.3 (Low reward)
            
            # Strict Genre Enforcement
            # If similarity is too low (e.g. < 0.1), giving ANY positive reward teaches the agent 
            # that "Quality > Genre". We want "Genre > Quality".
            
            if similarity < 0.1:
                final_reward = -1.0 # PUNITIVE Reward for mismatch
            else:
                 final_reward = (similarity * 2.0) + (avg_rating * 0.5)
            
        # Update State (Sliding Window)
        self.state_history.pop(0)
        self.state_history.append(movie_id)
        
        terminated = False
        truncated = False
        
        if self.steps_taken >= self.max_steps:
             truncated = True
        
        # If we hit the EXACT target movie, high reward and end?
        if movie_id == self.target_movie_id:
            final_reward += 5 # Bonus
            terminated = True
            
        return self._get_state(), final_reward, terminated, truncated, {}

    def render(self, mode='human'):
        print(f"Step: {self.steps_taken}, User: {self.current_user}")
        print(f"Target Context: {self.target_context}")
