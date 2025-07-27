import torch
import os
from sklearn.metrics import roc_auc_score, accuracy_score
import json
from datetime import datetime
from .model import SimpleGraphSAGE

def evaluate_model(data, model_path):
    """Evaluate the trained GraphSAGE model."""
    print("Starting model evaluation...")

    # ✅ CRITICAL FIX: Handle None model_path from terminated training
    if model_path is None:
        model_path = '/opt/airflow/models/best_graphsage_model.pth'
        print(f"⚠️ XCom model_path is None, using default: {model_path}")
    else:
        print(f"✅ XCom model_path found: {model_path}")
    
    # Verify file exists
    if not os.path.exists(model_path):
        # Try alternative paths
        alternative_paths = [
            '/opt/airflow/models/best_graphsage_model.pth',
            '/opt/airflow/models/graphsage_model.pth',
            '/opt/airflow/models/model.pth'
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"✅ Found model at alternative path: {model_path}")
                break
        else:
            raise FileNotFoundError(f"No model found at {model_path} or alternative paths")
    
    print(f"📁 Loading model from: {model_path}")
    
    # Your existing tensor processing (which is working perfectly!)
    u_feats = torch.tensor(data['user_features'], dtype=torch.float)
    m_feats = torch.tensor(data['movie_features'], dtype=torch.float)
    
    print(f"📊 User features shape: {u_feats.shape}")
    print(f"📊 Movie features shape: {m_feats.shape}")
    
    # Tensor padding (working correctly)
    padding_size = m_feats.shape[1] - u_feats.shape[1]  # 21 - 4 = 17
    user_padding = torch.zeros(u_feats.shape[0], padding_size)
    u_feats_padded = torch.cat([u_feats, user_padding], dim=1)
    
    print(f"✅ Padded user features shape: {u_feats_padded.shape}")
    
    x = torch.cat([u_feats_padded, m_feats], dim=0)
    num_users = len(u_feats_padded)
    
    print(f"📊 Combined features shape: {x.shape}")
    print(f"👥 Number of users: {num_users}")

    def make_edges(pos, neg):
        edges = pos + neg
        ei = torch.tensor([[e['user_id_mapped'] for e in edges],
                          [e['movie_id_mapped'] + num_users for e in edges]], dtype=torch.long)
        lbl = torch.tensor([1]*len(pos) + [0]*len(neg), dtype=torch.float)
        return ei, lbl

    splits = data['splits']
    base_edge_index, _ = make_edges(splits['train_pos'], splits['train_neg'])
    test_ei, test_lbl = make_edges(splits['test_pos'], splits['test_neg'])

    # 3. load model and evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleGraphSAGE(x.size(1), hidden_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        preds = torch.sigmoid(model(x.to(device),
                                   base_edge_index.to(device),
                                   test_ei.to(device))).cpu().numpy()

    test_auc = roc_auc_score(test_lbl.numpy(), preds)
    test_acc = accuracy_score(test_lbl.numpy(), (preds > 0.5).astype(int))

    # 4. Save metrics to file - CHANGED LOCATION
    evaluation_results = {
        "model_path": model_path,
        "test_auc": float(test_auc),
        "test_accuracy": float(test_acc),
        "evaluated_at": datetime.now().isoformat(),
        "device_used": str(device),
        "model_architecture": "GraphSAGE",
        "hidden_dim": 128,
        "test_samples": len(test_lbl),
        "positive_samples": int(test_lbl.sum()),
        "negative_samples": int(len(test_lbl) - test_lbl.sum())
    }
    
    # 📁 CHANGED: Save to results directory for better accessibility
    results_dir = "/opt/airflow/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save to JSON file
    results_file = os.path.join(results_dir, "evaluation_results.json")
    
    # If file exists, load existing results and append
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
            if not isinstance(existing_results, list):
                existing_results = [existing_results]
        except (json.JSONDecodeError, FileNotFoundError):
            existing_results = []
    else:
        existing_results = []
    
    # Append current results
    existing_results.append(evaluation_results)
    
    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(existing_results, f, indent=2)
    
    print(f"📄 Evaluation results saved to: {results_file}")
    print(f"🎯 Test AUC: {test_auc:.4f}")
    print(f"🎯 Test Accuracy: {test_acc:.4f}")
    
    # Also save a human-readable summary
    summary_file = os.path.join(results_dir, "evaluation_summary.txt")
    with open(summary_file, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Evaluation completed at: {evaluation_results['evaluated_at']}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test AUC: {test_auc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Test samples: {len(test_lbl)} (Pos: {int(test_lbl.sum())}, Neg: {int(len(test_lbl) - test_lbl.sum())})\n")
        f.write(f"{'='*50}\n")

    return {
        "message": "evaluation-done",
        "test_auc": float(test_auc),
        "test_accuracy": float(test_acc),
        "results_file": results_file
    }
