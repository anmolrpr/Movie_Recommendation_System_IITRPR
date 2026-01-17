import pandas as pd
import numpy as np
import os

class MovieLens100k:
    def __init__(self, data_dir="data/ml-100k"):
        self.data_dir = data_dir
        self.users = None
        self.items = None
        self.ratings = None
        self.genres = [
            "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", 
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", 
            "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
        
        self._load_data()

    def _load_data(self):
        # Load Ratings
        # user id | item id | rating | timestamp
        ratings_path = os.path.join(self.data_dir, "u.data")
        self.ratings = pd.read_csv(
            ratings_path, 
            sep='\t', 
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            encoding='latin-1'
        )
        
        # Load Items (Movies)
        # movie id | movie title | release date | video release date | IMDb URL | genres...
        items_path = os.path.join(self.data_dir, "u.item")
        genre_cols = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", 
                      "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", 
                      "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        
        col_names = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_cols
        
        self.items = pd.read_csv(
            items_path, 
            sep='|', 
            names=col_names,
            encoding='latin-1'
        )
        
        # Create a dictionary for movie features (genres)
        self.movie_features = {}
        for idx, row in self.items.iterrows():
            # Get the genre vector (last 19 columns)
            features = row[genre_cols].values.astype(float)
            self.movie_features[row['movie_id']] = features
            
        # Calculate Global Average Ratings
        self.avg_ratings = self.ratings.groupby('movie_id')['rating'].mean().to_dict()

    def get_average_rating(self, movie_id):
        """Returns the global average rating for a movie (default 3.0)."""
        return self.avg_ratings.get(movie_id, 3.0)

    def get_user_history(self, user_id):
        """Returns chronological interactions for a user."""
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        user_ratings = user_ratings.sort_values('timestamp')
        return user_ratings

    def get_movie_features(self, movie_id):
        """Returns the genre vector for a movie."""
        return self.movie_features.get(movie_id, np.zeros(len(self.genres)))
    
    def get_all_movie_ids(self):
        return self.items['movie_id'].unique()
    
    def get_stats(self):
        return {
            "num_users": self.ratings['user_id'].nunique(),
            "num_items": self.items['movie_id'].nunique(),
            "num_ratings": len(self.ratings)
        }

class MovieLens1M:
    def __init__(self, data_dir="data/ml-1m"):
        self.data_dir = data_dir
        self.ratings = None
        self.items = None
        self.genres = [
            "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", 
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", 
            "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
        # Note: ML-1M doesn't have "unknown", so list is shorter (18)
        
        self._load_data()

    def _load_data(self):
        # 1. Load Ratings (UserID::MovieID::Rating::Timestamp)
        ratings_path = os.path.join(self.data_dir, "ratings.dat")
        self.ratings = pd.read_csv(
            ratings_path, 
            sep='::', 
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python',
            encoding='latin-1'
        )
        
        # 2. Load Movies (MovieID::Title::Genres)
        # Genres are pipe-separated "Animation|Children's|Comedy"
        movies_path = os.path.join(self.data_dir, "movies.dat")
        self.items = pd.read_csv(
            movies_path, 
            sep='::', 
            names=['movie_id', 'title_year', 'genres_str'], # 3 cols
            engine='python',
            encoding='latin-1'
        )
        
        # Extract Title and Year
        # Titles are like "Toy Story (1995)"
        # We can extract year with regex or simple string split
        self.items['title'] = self.items['title_year'].apply(lambda x: x[:-7])
        self.items['release_date'] = self.items['title_year'].apply(lambda x: x[-5:-1]) # Just the year
        self.items['imdb_url'] = "#" # 1M doesn't have IMDb links
        
        # 3. Build Genre Features
        self.movie_features = {}
        
        # One-hot encode genres
        # Init zero columns
        for g in self.genres:
            self.items[g] = 0.0
            
        for idx, row in self.items.iterrows():
            my_genres = row['genres_str'].split('|')
            vec = np.zeros(len(self.genres))
            for g_name in my_genres:
                if g_name in self.genres:
                    g_idx = self.genres.index(g_name)
                    vec[g_idx] = 1.0
                    self.items.at[idx, g_name] = 1.0 # Store in DF too
            
            self.movie_features[row['movie_id']] = vec

        # 4. Calculate Global Average Ratings
        self.avg_ratings = self.ratings.groupby('movie_id')['rating'].mean().to_dict()

    def get_average_rating(self, movie_id):
        return self.avg_ratings.get(movie_id, 3.0)

    def get_user_history(self, user_id):
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        user_ratings = user_ratings.sort_values('timestamp')
        return user_ratings

    def get_movie_features(self, movie_id):
        return self.movie_features.get(movie_id, np.zeros(len(self.genres)))
    
    def get_all_movie_ids(self):
        return self.items['movie_id'].unique()
    
    def get_stats(self):
        return {
            "num_users": self.ratings['user_id'].nunique(),
            "num_items": self.items['movie_id'].nunique(),
            "num_ratings": len(self.ratings)
        }


if __name__ == "__main__":
    # Test
    ml = MovieLens100k()
    print(ml.get_stats())
    print("Example Movie Features (ID 1):", ml.get_movie_features(1))
