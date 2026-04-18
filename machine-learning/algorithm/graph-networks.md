# Graph Neural Networks

Graph-structured data — social networks, knowledge graphs, molecular structures, recommendation systems — cannot be processed by standard CNNs or RNNs because the topology is irregular. Graph Neural Networks (GNNs) operate directly on graph structure.

---

## Core Concepts

A graph `G = (V, E)` consists of:
- **Nodes (V)** — entities (users, products, atoms, documents)
- **Edges (E)** — relationships between entities (friendship, purchase, bond, citation)
- **Node features** — attribute vectors attached to each node
- **Edge features** — attribute vectors attached to each edge (optional)
- **Graph-level properties** — labels or features for the whole graph

```
Node: user_id=42, features=[age, tenure, spend]
Edge: (user_42, product_17), features=[purchase_count, last_seen_days]
```

---

## Message Passing — The GNN Core

Most GNNs follow a message passing framework: each node aggregates information from its neighbours, updates its own representation, and repeats for L layers.

```
For each layer l:
  For each node v:
    messages = AGGREGATE({ h_u^(l-1) : u ∈ neighbours(v) })
    h_v^(l) = UPDATE(h_v^(l-1), messages)
```

After L layers, each node's representation encodes information from its L-hop neighbourhood.

---

## Graph Embedding Methods

### Matrix Factorisation

**Core idea:** factorise the adjacency matrix (or a derived matrix like the co-occurrence matrix) into low-dimensional node embeddings. Nodes that co-occur frequently in the graph end up close in embedding space.

```
A ≈ Z · Z^T    where Z ∈ R^(N × d) are the node embeddings
```

| Pros | Cons |
|------|------|
| Simple, well-understood | Does not generalise to unseen nodes (transductive) |
| Efficient for small graphs | Does not use node features |
| Good for recommendation (user-item matrix) | Scalability issues for very large graphs |

**When to use:** collaborative filtering, link prediction on static graphs where all nodes are known at training time.

---

### Node2Vec

**Core idea:** learn node embeddings by running biased random walks on the graph and applying Word2Vec (Skip-gram) to the resulting node sequences. Nodes that appear in similar walk contexts get similar embeddings.

**Biased random walk:** two parameters control the walk behaviour:
- `p` (return parameter) — probability of returning to the previous node. Low p = DFS-like (explores deep paths)
- `q` (in-out parameter) — probability of moving to a node far from the previous. Low q = BFS-like (explores local neighbourhood)

```
Walk: v1 → v2 → v3 → v4 → ...
  At each step, transition probabilities are biased by p and q
  → Apply Skip-gram on walks → node embeddings
```

| Pros | Cons |
|------|------|
| Captures both local structure (BFS) and global structure (DFS) via p/q | Transductive — does not generalise to new nodes |
| No node features required | Random walks are stochastic — results vary between runs |
| Scalable to large graphs | Does not use edge features |

**When to use:** node classification, link prediction, community detection on graphs without rich node features.

---

### Graph Convolutional Networks (GCN)

**Core idea:** generalise convolution to graphs. Each node aggregates the mean of its neighbours' features (weighted by degree), then applies a learnable linear transformation.

**Layer update rule:**
```
H^(l+1) = σ( D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l) )
```
- `Ã = A + I` — adjacency matrix with self-loops added
- `D̃` — degree matrix of `Ã`
- `H^(l)` — node feature matrix at layer l
- `W^(l)` — learnable weight matrix
- `σ` — activation function (ReLU)

The symmetric normalisation `D̃^(-1/2) Ã D̃^(-1/2)` prevents high-degree nodes from dominating the aggregation.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    """
    Two-layer Graph Convolutional Network for node classification.

    Parameters
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden layer dimension.
    out_channels : int
        Number of output classes.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)
```

| Pros | Cons |
|------|------|
| Uses node features — richer than Node2Vec | Transductive in basic form |
| Inductive variants (GraphSAGE) generalise to new nodes | Over-smoothing — deep GCNs make all node representations similar |
| Strong on citation networks, molecular graphs | Sensitive to graph structure quality |

---

### GraphSAGE (Inductive)

**Core idea:** instead of using the full graph adjacency, sample a fixed-size neighbourhood for each node and aggregate sampled neighbours. This makes the model inductive — it can generate embeddings for nodes not seen during training.

```
For each node v:
  Sample k neighbours N_k(v)
  h_v = AGGREGATE({ h_u : u ∈ N_k(v) })
  h_v = σ( W · CONCAT(h_v^(l-1), h_v) )
```

Aggregators: mean, LSTM, max-pooling.

**When to use:** large graphs where full-graph training is infeasible, or when new nodes are added at inference time (e.g. new users in a recommendation system).

---

### Graph AutoEncoder (GAE)

**Core idea:** encode node features into embeddings with a GNN encoder, then decode by reconstructing the adjacency matrix (predict which edges exist).

```
Encoder: Z = GCN(X, A)          → node embeddings
Decoder: Â = σ(Z · Z^T)         → reconstructed adjacency
Loss:    BCE(A, Â)               → binary cross-entropy on edge existence
```

**When to use:** link prediction, graph generation, unsupervised node embedding learning.

---

## GNN Tasks

| Task | Description | Example |
|------|-------------|---------|
| Node classification | Predict a label for each node | Classify users as churned/active |
| Link prediction | Predict whether an edge exists | Recommend products to users |
| Graph classification | Predict a label for the whole graph | Classify molecules as toxic/non-toxic |
| Node clustering | Group nodes into communities | Detect communities in social networks |
| Graph generation | Generate new graphs with desired properties | Drug discovery — generate valid molecules |

---

## Off-Policy Evaluation

Off-policy evaluation (OPE) estimates the performance of a target policy using data collected under a different (behaviour) policy — without deploying the target policy.

**Why it matters:** in recommendation and pricing systems, you cannot A/B test every candidate policy. OPE lets you estimate the value of a new policy from logged historical data.

### Inverse Propensity Scoring (IPS)

Re-weight logged rewards by the ratio of target policy probability to behaviour policy probability:

```
V_IPS(π) = (1/N) Σ [ π(a|x) / π_b(a|x) ] · r
```

- `π(a|x)` — probability of action `a` under the target policy
- `π_b(a|x)` — probability of action `a` under the behaviour policy (logged)
- `r` — observed reward

High variance when `π / π_b` is large — the target policy takes actions the behaviour policy rarely took.

### Doubly Robust (DR) Estimator

Combines a direct model estimate with IPS correction:

```
V_DR(π) = (1/N) Σ [ q̂(x, a) + (π/π_b) · (r - q̂(x, a)) ]
```

- `q̂(x, a)` — learned reward model (direct method)
- The IPS term corrects for bias in the reward model

DR is unbiased if either the reward model or the propensity scores are correct — doubly robust.

| Method | Bias | Variance | Requires |
|--------|------|----------|---------|
| Direct Method (DM) | High if model is wrong | Low | Reward model |
| IPS | Low | High if policies differ | Propensity scores |
| Doubly Robust (DR) | Low | Medium | Both |
