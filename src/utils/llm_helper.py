import numpy as np
import os
import google.generativeai as genai
import json
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    def __init__(self):
        # ML-1M Genres (18) - Removed "unknown"
        self.genres = [
            "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", 
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", 
            "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
        
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if self.api_key and "YOUR_GOOGLE_API_KEY" not in self.api_key:
            genai.configure(api_key=self.api_key)
            # Switching to google/gemma-3-27b-it as requested
            self.model = genai.GenerativeModel('google/gemma-3-27b-it')
            self.use_real_llm = True
        else:
            print("Warning: GOOGLE_API_KEY not set or is placeholder. Using Mock LLM.")
            self.use_real_llm = False
            
    def get_mood_features(self, user_text):
        """
        Extracts genres from text using Gemini (or fallback).
        Returns a multi-hot vector of genres.
        """
        if self.use_real_llm:
            return self._call_gemini(user_text)
        else:
            return self._mock_extraction(user_text)

    def _call_gemini(self, user_text):
        features = np.zeros(len(self.genres))
        
        prompt = f"""
        Analyze the following user mood/request for a movie recommendation: "{user_text}"
        
        Map this mood to the following exact list of genres:
        {json.dumps(self.genres)}
        
        Return ONLY a JSON array of strings containing the matching genres. 
        Example Output: ["Drama", "Romance"]
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Simple cleanup to ensure we get just the JSON
            text_resp = response.text.strip()
            if text_resp.startswith("```json"):
                text_resp = text_resp[7:-3]
            elif text_resp.startswith("```"):
                text_resp = text_resp[3:-3]
                
            predicted_genres = json.loads(text_resp)
            
            # Map back to vector
            for g in predicted_genres:
                if g in self.genres:
                    idx = self.genres.index(g)
                    features[idx] = 1.0
                    
            return features
            
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return self._mock_extraction(user_text)

    def _mock_extraction(self, user_text):
        # Fallback to simple keyword matching
        user_text = user_text.lower()
        features = np.zeros(len(self.genres))
        
        # Enhanced mapping
        keyword_map = {
            "funny": "Comedy",
            "comedy": "Comedy",
            "scary": "Horror",
            "horror": "Horror",
            "sad": "Drama",
            "drama": "Drama",
            "cry": "Drama",
            "action": "Action",
            "fight": "Action",
            "love": "Romance",
            "romance": "Romance",
            "space": "Sci-Fi",
            "future": "Sci-Fi",
            "sci-fi": "Sci-Fi",
            "kids": "Children's",
            "cartoon": "Animation"
        }
        
        found = False
        for i, genre in enumerate(self.genres):
            # 1. Direct match
            if genre.lower() in user_text:
                features[i] = 1.0
                found = True
        
        # 2. Keyword Map
        if not found:
             for key, val in keyword_map.items():
                 if key in user_text:
                     if val in self.genres:
                         idx = self.genres.index(val)
                         features[idx] = 1.0
                         found = True
        
        if not found:
            # Random genre if totally unknown
            random_idx = np.random.randint(len(self.genres))
            features[random_idx] = 1.0
            
        return features

if __name__ == "__main__":
    llm = LLMService()
    # Test (will likely use mock if no key)
    print("Vector:", llm.get_mood_features("I want a space adventure"))
