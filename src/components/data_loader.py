import pandas as pd

def load_raw_data():
    """Load raw data files from the mounted data directory."""
    df_ratings = pd.read_csv("/opt/airflow/data/u.data", sep="\t", header=None,
                             names=["user_id", "item_id", "rating", "timestamp"])
    df_users = pd.read_csv("/opt/airflow/data/u.user", sep="|", header=None,
                           names=["user_id", "age", "gender", "occupation", "zip_code"])
    
    item_cols = ["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL",
                 "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
                 "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                 "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    
    df_items = pd.read_csv("/opt/airflow/data/u.item", sep="|", header=None, 
                          names=item_cols, encoding="latin-1")

    # Map user/movie IDs
    user_id_map = {raw: idx for idx, raw in enumerate(df_users["user_id"])}
    movie_id_map = {raw: idx for idx, raw in enumerate(df_items["movie_id"])}

    df_ratings["user_id_mapped"] = df_ratings["user_id"].map(user_id_map)
    df_ratings["movie_id_mapped"] = df_ratings["item_id"].map(movie_id_map)

    return {
        'users': df_users.to_dict(),
        'items': df_items.to_dict(), 
        'ratings': df_ratings.to_dict(),
        'item_cols': item_cols
    }
