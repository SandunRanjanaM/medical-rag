import numpy as np
import torch
import torch.nn as nn
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

# --------------------
# Utility
# --------------------
def l2_normalize(vec: np.ndarray):
    return vec / np.linalg.norm(vec)

# --------------------
# InBEDDER
# --------------------
inbedder = SentenceTransformersDocumentEmbedder(
    model="KomeijiForce/inbedder-roberta-large",
    progress_bar=False,
)
inbedder.warm_up()

def embed_user_query(query: str):
    instruction = "Cluster this cardiology question semantically:"
    doc = Document(content=f"{instruction}\n{query}")
    result = inbedder.run(documents=[doc])
    return l2_normalize(np.array(result["documents"][0].embedding))

# --------------------
# Projection Layer
# --------------------
class ProjectionLayer(nn.Module):
    def __init__(self, in_dim=1024, out_dim=768):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x):
        return self.proj(x)

def project_user_query(projector, inbed_emb):
    with torch.no_grad():
        x = torch.tensor(inbed_emb, dtype=torch.float32).unsqueeze(0)
        projected = projector(x).squeeze(0).numpy()
    return projected
