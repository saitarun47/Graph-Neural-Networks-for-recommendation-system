import pandas as pd
import random
from sklearn.model_selection import train_test_split

def create_train_val_test_splits(processed_data):
    """Split data into train/validation/test sets with negative sampling."""
    # Reconstructing the ratings data
    ratings_data = processed_data['ratings_data']
    df_ratings = pd.DataFrame({
        'user_id_mapped': ratings_data['user_id_mapped'],
        'movie_id_mapped': ratings_data['movie_id_mapped']
    })

    num_users = len(processed_data['user_features'])
    num_movies = len(processed_data['movie_features'])

    # Splitting positive edges
    pos_df = df_ratings[["user_id_mapped","movie_id_mapped"]]
    train_val, test_df = train_test_split(pos_df, test_size=0.10, random_state=42)
    train_df, val_df = train_test_split(train_val, test_size=0.1111, random_state=42)

    # Generating negative samples
    all_pos = set(zip(pos_df.user_id_mapped, pos_df.movie_id_mapped))

    def sample_neg(n):
        negs = set()
        while len(negs) < n:
            u = random.randrange(num_users)
            v = random.randrange(num_movies)
            if (u,v) not in all_pos:
                negs.add((u,v))
        return pd.DataFrame(list(negs), columns=["user_id_mapped","movie_id_mapped"])
        
    neg_train = sample_neg(len(train_df))
    neg_val = sample_neg(len(val_df))
    neg_test = sample_neg(len(test_df))
    
    return {
        'user_features': processed_data['user_features'],
        'movie_features': processed_data['movie_features'],
        'splits': {
            'train_pos': train_df.to_dict('records'),
            'train_neg': neg_train.to_dict('records'),
            'val_pos': val_df.to_dict('records'),
            'val_neg': neg_val.to_dict('records'),
            'test_pos': test_df.to_dict('records'),
            'test_neg': neg_test.to_dict('records')
        }
    }
