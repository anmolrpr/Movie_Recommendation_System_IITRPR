import torch
import numpy as np
from src.agent.dqn import DQNAgent
from src.data.movielens import MovieLens1M
from src.utils.llm_helper import LLMService
from src.models import Interaction, User
from beanie.operators import In
import pandas as pd

class RecommenderService:
    def __init__(self):
        self.data_loader = None
        self.agent = None
        self.llm_service = None
        self.history_len = 5
        self.genre_dim = 18 # Updated
        self.ids = None
        
    def initialize(self, model_path="models/dqn_rec.pth"):
        print("Initializing Recommender Service...")
        # 1. Load Static Data
        self.data_loader = MovieLens1M()
        self.llm_service = LLMService()
        
        # 2. Setup Dimensions
        self.movie_ids = self.data_loader.get_all_movie_ids()
        self.movie_id_to_index = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.index_to_movie_id = {i: mid for mid, i in self.movie_id_to_index.items()}
        
        state_dim = (self.history_len * self.genre_dim) + self.genre_dim
        action_dim = len(self.movie_ids)
        
        # 3. Load Agent
        self.agent = DQNAgent(state_dim, action_dim)
        try:
            self.agent.load(model_path)
            self.agent.q_network.eval() # Set to eval mode
            print(f"Agent loaded from {model_path}")
        except FileNotFoundError:
            print("Warning: Model file not found. Using random agent.")

    async def get_recommendation(self, user: User, mood_text: str):
        # 1. Fetch User History from DB (Beanie)
        # Fetch ALL interactions for exclusion to ensure strict no-repeat
        all_interactions = await Interaction.find(
            Interaction.user_id == str(user.id)
        ).sort("-timestamp").to_list()
        
        # Note: interactions are newest first, we need oldest -> newest for the sequence
        # We want the 5 newest for state.
        recent_interactions = all_interactions[:self.history_len]
        recent_interactions.reverse() # Now Oldest -> Newest
        
        # 2. Build History Vector
        history_vecs = []
        for interaction in recent_interactions:
            # We need the genre features for this movie
            feat = self.data_loader.get_movie_features(interaction.movie_id)
            history_vecs.append(feat)
            
        # Pad with zeros if short
        while len(history_vecs) < self.history_len:
            history_vecs.insert(0, np.zeros(self.genre_dim))
            
        history_flat = np.concatenate(history_vecs)
        
        # 3. Build Context (Mood)
        context_vector = self.llm_service.get_mood_features(mood_text)
        
        # 4. Concatenate State
        state = np.concatenate([history_flat, context_vector]).astype(np.float32)
        
        # 5. Agent Action with Top-K Sampling & Exclusion
        # Get raw Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
        with torch.no_grad():
            q_values = self.agent.q_network(state_tensor) # shape [1, 1682]
            
        # exclude movies already in history from this session (last 20 to be safe)
        excluded_ids = [i.movie_id for i in all_interactions] 
        print(f"DEBUG: Excluding {len(excluded_ids)} movies: {excluded_ids[:10]}...")

        # Mask out excluded movies by setting Q-value to -inf
        for mid in excluded_ids:
            if mid in self.movie_id_to_index:
                idx = self.movie_id_to_index[mid]
                q_values[0, idx] = -float('inf')

        # Top-K Sampling (K=10) with Temperature
        # Temperature > 1.0 makes it more random, < 1.0 makes it more greedy
        temperature = 3.0 # Increase temperature for more variety
        
        scaled_q = q_values / temperature
        top_k = 50 # Increase candidate pool significantly
        top_probs, top_indices = torch.topk(torch.softmax(scaled_q, dim=1), top_k)
        
        # Convert to numpy for random choice
        probs = top_probs.cpu().numpy().flatten()
        probs = probs / probs.sum() # Renormalize
        indices = top_indices.cpu().numpy().flatten()
        
        # Sample a larger pool to Ensure we can fill 5 slots after filtering
        candidate_pool_size = 100 # Check up to 100 candidates
        if len(indices) < candidate_pool_size:
             sampled_indices = indices
        else:
             sampled_indices = np.random.choice(indices, size=candidate_pool_size, replace=False, p=probs)
        
        results = []
        target_count = 5
        
        for idx in sampled_indices:
            if len(results) >= target_count:
                break
                
            mid = self.index_to_movie_id[idx]
            # Retrieve details
            movie_row = self.data_loader.items[self.data_loader.items['movie_id'] == mid]
            if not movie_row.empty:
                row = movie_row.iloc[0]
                title = row['title']
                release_date = str(row['release_date'])
                imdb_url = str(row['imdb_url'])
                
                # Get Genres
                active_genres = []
                for genre in self.data_loader.genres:
                    if genre in row and row[genre] == 1:
                        active_genres.append(genre)
                genres_str = ", ".join(active_genres)
                
            else:
                title = f"Movie {mid}"
                release_date = "Unknown"
                imdb_url = "#"
                genres_str = "Unknown"
            
            # Calculate match score for display
            score = float(np.dot(self.data_loader.get_movie_features(mid), context_vector))
            
            # Use strict filtering: only include if score > 0 (at least some genre match)
            if score > 0.01:
                results.append({
                    "movie_id": int(mid),
                    "title": title,
                    "release_date": release_date,
                    "imdb_url": imdb_url,
                    "genres": genres_str,
                    "genre_match_score": score
                })
        
        if len(results) < target_count:
            print(f"Warning: Only found {len(results)} matches. Triggering Fallback.")
            # Fallback: Search DB directly for specific genre matches
            # 1. Identify Target Genres from context_vector
            target_indices = np.where(context_vector > 0)[0]
            target_genres = [self.data_loader.genres[i] for i in target_indices]
            
            if target_genres:
                print(f"Fallback searching for: {target_genres}")
                # 2. Filter dataset
                # We want movies that have 1 for ANY of the target genres
                # This equals: (df[g1] == 1) | (df[g2] == 1)...
                
                # Start with all items
                candidates = self.data_loader.items
                
                # Filter mask
                mask = False
                for g in target_genres:
                    if g in candidates.columns:
                        mask = mask | (candidates[g] == 1)
                        
                if isinstance(mask, bool) and not mask:
                     # No genres found in columns? Should not happen
                     filtered = pd.DataFrame() 
                else:
                    filtered = candidates[mask]
                
                # Exclude already recommended
                current_ids = [r['movie_id'] for r in results] + excluded_ids
                filtered = filtered[~filtered['movie_id'].isin(current_ids)]
                
                # Sample random fill
                needed = target_count - len(results)
                if not filtered.empty:
                    # Take sample
                    n_sample = min(needed, len(filtered))
                    api_fallback_samples = filtered.sample(n=n_sample)
                    
                    for _, row in api_fallback_samples.iterrows():
                        mid = row['movie_id']
                        # Add to results
                        active_genres = []
                        for genre in self.data_loader.genres:
                            if genre in row and row[genre] == 1:
                                active_genres.append(genre)
                                
                        results.append({
                            "movie_id": int(mid),
                            "title": row['title'],
                            "release_date": str(row['release_date']),
                            "imdb_url": str(row['imdb_url']),
                            "genres": ", ".join(active_genres),
                            "genre_match_score": 0.99 # Fake high score for fallback
                        })
                        
        return results
            
recommender = RecommenderService()
