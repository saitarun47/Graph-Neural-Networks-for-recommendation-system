import sys
sys.path.append('/opt/airflow/src')

import pandas as pd
import gc
import torch
from components.data_loader import load_raw_data
from components.feature_engineering import create_user_features, create_movie_features
from components.graph_builder import build_heterogeneous_graph
from components.data_splitter import create_train_val_test_splits
from components.neo4j_storage import store_data_in_neo4j
from components.trainer import train_model
from components.evaluator import evaluate_model

def load_data():
    """Airflow task: Load raw data."""
    return load_raw_data()

def load_and_process(**context):
    """Airflow task: Process features."""
    import gc
    # Get data from previous task
    data_dict = context['ti'].xcom_pull(task_ids='gnn_etl')

    # Reconstruct dataframes
    df_users = pd.DataFrame.from_dict(data_dict['users'])
    df_items = pd.DataFrame.from_dict(data_dict['items'])
    df_ratings = pd.DataFrame.from_dict(data_dict['ratings'])
    item_cols = data_dict['item_cols']
    
    u_feats = create_user_features(df_users, df_ratings)
    m_feats = create_movie_features(df_items, df_ratings, item_cols)

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Return serializable data for next task
    return {
        'user_features': u_feats.tolist(),
        'movie_features': m_feats.tolist(),
        'ratings_data': {
            'user_id_mapped': df_ratings["user_id_mapped"].tolist(),
            'movie_id_mapped': df_ratings["movie_id_mapped"].tolist()
        }
    }

def build_graph(**context):
    """Airflow task: Build graph structure."""
    # Get processed data from previous task
    processed_data = context['ti'].xcom_pull(task_ids='gnn_etl2')
    return build_heterogeneous_graph(processed_data)

def split_and_sample(**context):
    """Airflow task: Create train/val/test splits."""
    # Get graph data from previous task
    processed_data = context['ti'].xcom_pull(task_ids='gnn_etl2')
    return create_train_val_test_splits(processed_data)

def store_in_neo4j(**context):
    """Airflow task: Store data in Neo4j."""
    data = context['ti'].xcom_pull(task_ids='gnn_etl4')
    return store_data_in_neo4j(data)

def train_gnn_model(**context):
    """Airflow task: Train the GNN model."""
    data = context['ti'].xcom_pull(task_ids='gnn_etl4')
    return train_model(data, context)

def evaluate_gnn_model(**context):
    """Airflow task: Evaluate the trained model."""
    data = context['ti'].xcom_pull(task_ids='gnn_etl4')
    model_path = context['ti'].xcom_pull(task_ids='gnn_train', key='model_path')
    return evaluate_model(data, model_path)
