import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import os
from .model import SimpleGraphSAGE

def train_model(data, context):
    """Train the GraphSAGE model with the provided data."""
    print("Starting GraphSAGE training...")
    
    # Reconstruct graph data for training
    u_feats = torch.tensor(data['user_features'], dtype=torch.float)
    m_feats = torch.tensor(data['movie_features'], dtype=torch.float)

    print(f"User features shape: {u_feats.shape}")
    print(f"Movie features shape: {m_feats.shape}")

   
    padding_size = m_feats.shape[1] - u_feats.shape[1]  
    user_padding = torch.zeros(u_feats.shape[0], padding_size)
    u_feats_padded = torch.cat([u_feats, user_padding], dim=1)
    
    print(f"Padded user features shape: {u_feats_padded.shape}")

   
    x = torch.cat([u_feats_padded, m_feats], dim=0)
    num_users = len(u_feats_padded)
    
    print(f"Combined features shape: {x.shape}")
    print(f"Number of users: {num_users}")
    
    # Build edge tensors
    def build_edge_tensors(pos_edges, neg_edges):
        all_edges = pos_edges + neg_edges
        edge_index = torch.tensor([
            [edge['user_id_mapped'] for edge in all_edges],
            [edge['movie_id_mapped'] + num_users for edge in all_edges]  
        ], dtype=torch.long)
        labels = torch.tensor([1] * len(pos_edges) + [0] * len(neg_edges), dtype=torch.float)
        return edge_index, labels
    
    splits = data['splits']
    
    # Create all edge data
    train_ei, train_lbl = build_edge_tensors(splits['train_pos'], splits['train_neg'])
    val_ei, val_lbl = build_edge_tensors(splits['val_pos'], splits['val_neg'])
    test_ei, test_lbl = build_edge_tensors(splits['test_pos'], splits['test_neg'])
    
    # Create base graph (all training edges for message passing)
    base_edge_index = train_ei
    
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleGraphSAGE(x.size(1), hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # Move data to device
    x = x.to(device)
    base_edge_index = base_edge_index.to(device)
    train_ei = train_ei.to(device)
    train_lbl = train_lbl.to(device)
    val_ei = val_ei.to(device)
    val_lbl = val_lbl.to(device)
    
    best_val_auc = 0
    patience_counter = 0
    patience = 20
    
    print("Training started...")
    
    for epoch in range(1, 1000):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(x, base_edge_index, train_ei)
        loss = loss_fn(out, train_lbl)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(x, base_edge_index, val_ei)
                val_loss = loss_fn(val_out, val_lbl)
                
                val_pred = torch.sigmoid(val_out).cpu().numpy()
                val_auc = roc_auc_score(val_lbl.cpu().numpy(), val_pred)
                
                print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    best_model = model.state_dict().copy()
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch}")
                    break
    
    # Load best model and final evaluation
    model.load_state_dict(best_model)
    model.eval()
    
    with torch.no_grad():
        test_out = model(x, base_edge_index, test_ei.to(device))
        test_pred = torch.sigmoid(test_out).cpu().numpy()
        test_auc = roc_auc_score(test_lbl.numpy(), test_pred)
        test_acc = ((test_pred > 0.5) == test_lbl.numpy()).mean()
    
    print(f"Final Results:")
    print(f"Best Val AUC: {best_val_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save model
    model_path = "/opt/airflow/models/best_graphsage_model.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(best_model, model_path)
    
    # push artifact location to XCom for the next task
    context['ti'].xcom_push(key='model_path', value=model_path)

    return {
        "message": "training-done",
        "best_val_auc": float(best_val_auc)
    }
