import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv

# ──────────────── 1. LOAD AND PREPROCESS DATA ────────────────

# 1.1 Ratings
df_ratings = pd.read_csv(
    "u.data",
    sep="\t",                
    header=None,
    names=["user_id", "item_id", "rating", "timestamp"]
)

# 1.2 Users
df_users = pd.read_csv(
    "u.user",
    sep="|",
    header=None,
    names=["user_id", "age", "gender", "occupation", "zip_code"]
)

# 1.3 Items
item_cols = [
    "movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
df_items = pd.read_csv(
    "u.item",
    sep="|",
    header=None,
    names=item_cols,
    encoding="latin-1"
)

# ──────────────── 2. BUILD HETEROGENEOUS GRAPH ────────────────

# 2.1 Create 0-based mappings
user_id_map  = {raw: idx for idx, raw in enumerate(df_users["user_id"])}
movie_id_map = {raw: idx for idx, raw in enumerate(df_items["movie_id"])}

df_ratings["user_id_mapped"]  = df_ratings["user_id"].map(user_id_map)
df_ratings["movie_id_mapped"] = df_ratings["item_id"].map(movie_id_map)

num_users  = len(user_id_map)
num_movies = len(movie_id_map)

# 2.2 Build node features
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Users: scale age & encode occupation
occ_enc = LabelEncoder().fit(df_users["occupation"])
age_scaled = MinMaxScaler().fit_transform(df_users[["age"]])
occ_encoded = occ_enc.transform(df_users["occupation"])[:, None]
u_feats = torch.tensor(
    np.hstack([age_scaled, occ_encoded]), dtype=torch.float
)

# Movies: genre one‑hots
m_feats = torch.tensor(df_items[item_cols[5:]].values, dtype=torch.float)

# 2.3 Construct HeteroData
hetero = HeteroData()
hetero["user"].x  = u_feats
hetero["movie"].x = m_feats

edge_index = torch.tensor([
    df_ratings["user_id_mapped"].values,
    df_ratings["movie_id_mapped"].values
], dtype=torch.long)
hetero["user", "rates", "movie"].edge_index = edge_index

# 2.4 Convert to homogeneous graph
data = hetero.to_homogeneous(node_attrs=["x"], edge_attrs=None)
# Now: data.x  has shape [num_users+num_movies, feat_dim]
#      data.edge_index has shape [2, num_edges]

# ──────────────── 3. MANUAL LINK‑PREDICTION SPLIT ────────────────

# 3.1 Positive edges
pos_df = df_ratings[["user_id_mapped", "movie_id_mapped"]]

# 80/10/10 split
train_val, test_df = train_test_split(pos_df, test_size=0.10, random_state=42)
train_df,  val_df  = train_test_split(train_val, test_size=0.1111, random_state=42)

# 3.2 Sample strict negatives (exclude all real edges)
all_pos = set(zip(pos_df.user_id_mapped, pos_df.movie_id_mapped))
def sample_neg(n):
    negs = set()
    while len(negs) < n:
        u = random.randrange(num_users)
        v = random.randrange(num_movies)
        if (u, v) not in all_pos:
            negs.add((u, v))
    return pd.DataFrame(list(negs), columns=["user_id_mapped","movie_id_mapped"])

neg_train = sample_neg(len(train_df))
neg_val   = sample_neg(len(val_df))
neg_test  = sample_neg(len(test_df))

# 3.3 Build edge_index & labels and shift movie IDs
def build_edges(pos, neg):
    u_list = list(pos.user_id_mapped)  + list(neg.user_id_mapped)
    m_list = [m + num_users for m in list(pos.movie_id_mapped) + list(neg.movie_id_mapped)]
    ei = torch.tensor([u_list, m_list], dtype=torch.long)
    lbl = torch.tensor([1]*len(pos) + [0]*len(neg), dtype=torch.float)
    return ei, lbl

train_ei, train_lbl = build_edges(train_df, neg_train)
val_ei,   val_lbl   = build_edges(val_df,   neg_val)
test_ei,  test_lbl  = build_edges(test_df,  neg_test)

# ──────────────── 4. MODEL DEFINITION ────────────────

class GNNLinkPredict(torch.nn.Module):
    def __init__(self, in_feats, hidden_dim=64, decoder_dim=64):
        super().__init__()
        self.conv1 = GATConv(in_feats, hidden_dim, heads=2, concat=True)
        self.conv2 = GATConv(hidden_dim*2, hidden_dim, heads=2, concat=True)
        self.lin1  = torch.nn.Linear(2*hidden_dim*2, decoder_dim)
        self.lin2  = torch.nn.Linear(decoder_dim,     1)

    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        return self.conv2(h, edge_index)

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        h = torch.cat([z[src], z[dst]], dim=1)
        h = F.relu(self.lin1(h))
        return self.lin2(h).view(-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

# ──────────────── 5. TRAINING LOOP SKETCH ────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GNNLinkPredict(data.num_features, 128, 128).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn= torch.nn.BCEWithLogitsLoss()

x  = data.x.to(device)
ei = data.edge_index.to(device)

for epoch in range(1, 1000):
    model.train()
    opt.zero_grad()
    out = model(x, ei, train_ei.to(device))
    loss = loss_fn(out, train_lbl.to(device))
    loss.backward()
    opt.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_out  = model(x, ei, val_ei.to(device))
            val_loss = loss_fn(val_out, val_lbl.to(device))
        print(f"Epoch {epoch:02d} — train_loss: {loss:.4f}, val_loss: {val_loss:.4f}")

# Final test evaluation
model.eval()
with torch.no_grad():
    test_out  = model(x, ei, test_ei.to(device))
    test_loss = loss_fn(test_out, test_lbl.to(device))
print(f"Test loss: {test_loss:.4f}")
