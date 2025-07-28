import torch
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np

def build_heterogeneous_graph(raw_data):
    """Build a heterogeneous graph from raw data."""
    
   
    df_users = pd.DataFrame(raw_data['users'])
    df_items = pd.DataFrame(raw_data['items'])
    df_ratings = pd.DataFrame(raw_data['ratings'])
    
    # Genre columns (18 features)
    genre_columns = [
        "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", 
        "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", 
        "Thriller", "War", "Western"
    ]
    
    
    user_features = []
    for _, user in df_users.iterrows():
        # Basic features: [age_normalized, gender, occupation_encoded] + 18 zeros for padding
        age = user['age'] / 100.0  # Normalize age to 0-1
        gender = 1.0 if user['gender'] == 'M' else 0.0  # Binary encoding
        occupation = hash(str(user['occupation'])) % 16 / 16.0  # Normalize occupation hash
        
    
        feature_vector = [age, gender, occupation] + [0.0] * 18
        user_features.append(feature_vector)
    
   
    movie_features = []
    for _, movie in df_items.iterrows():
       
        placeholder_features = [0.0, 0.0, 0.0]  # 3 placeholders for user-like features
        genre_vector = [float(movie.get(genre, 0)) for genre in genre_columns]
        
        feature_vector = placeholder_features + genre_vector  
        movie_features.append(feature_vector)
    
    # Convert to tensors (both are 21-dimensional)
    u_feats = torch.tensor(user_features, dtype=torch.float)
    m_feats = torch.tensor(movie_features, dtype=torch.float)
    
    print(f"User features shape: {u_feats.shape}")    
    print(f"Movie features shape: {m_feats.shape}")   

    # Building PyTorch Geometric heterogeneous graph
    hetero = HeteroData()
    hetero["user"].x = u_feats
    hetero["movie"].x = m_feats

    # Creating edge index from ratings
    edge_index = torch.tensor([
        df_ratings['user_id_mapped'].tolist(),
        df_ratings['movie_id_mapped'].tolist()
    ], dtype=torch.long)
    
    hetero["user", "rates", "movie"].edge_index = edge_index
    
  
    return {
        'x': torch.cat([u_feats, m_feats], dim=0),  
        'edge_index': edge_index,
        'users': raw_data['users'],
        'items': raw_data['items'], 
        'ratings': raw_data['ratings'],
        'hetero_data': hetero
    }