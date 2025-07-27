import torch
from torch_geometric.data import HeteroData

def build_heterogeneous_graph(processed_data):
    """Build a heterogeneous graph from user and movie features."""
    # Reconstruct tensors
    u_feats = torch.tensor(processed_data['user_features'], dtype=torch.float)
    m_feats = torch.tensor(processed_data['movie_features'], dtype=torch.float)

    hetero = HeteroData()
    hetero["user"].x = u_feats
    hetero["movie"].x = m_feats

    edge_index = torch.tensor([
        processed_data['ratings_data']['user_id_mapped'],
        processed_data['ratings_data']['movie_id_mapped']
    ], dtype=torch.long)
    hetero["user", "rates", "movie"].edge_index = edge_index

    return "Graph built successfully"
