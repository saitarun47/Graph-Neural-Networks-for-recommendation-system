import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def create_user_features(df_users, df_ratings):
    """Create feature vectors for users."""
    age_scaled = MinMaxScaler().fit_transform(df_users[["age"]])
    occ_enc = LabelEncoder().fit(df_users["occupation"])
    occ_encoded = occ_enc.transform(df_users["occupation"])[:, None]
    
    user_stats = df_ratings.groupby('user_id').agg({
        'rating': ['mean', 'count']
    }).fillna(0)
    user_stats.columns = ['avg_rating', 'num_ratings']
    stats_scaled = MinMaxScaler().fit_transform(user_stats.values)
    
    features = np.hstack([age_scaled, occ_encoded, stats_scaled])
    return torch.tensor(features, dtype=torch.float)

def create_movie_features(df_items, df_ratings, item_cols):
    """Create feature vectors for movies."""
    genre_features = df_items[item_cols[5:]].values
    
    movie_stats = df_ratings.groupby('item_id').agg({
        'rating': ['mean', 'count']
    }).fillna(0)
    movie_stats.columns = ['avg_rating', 'num_ratings']
    stats_scaled = MinMaxScaler().fit_transform(movie_stats.values)
    
    features = np.hstack([genre_features, stats_scaled])
    return torch.tensor(features, dtype=torch.float)
