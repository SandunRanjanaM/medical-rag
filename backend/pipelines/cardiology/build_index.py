# backend/pipelines/cardiology/build_index.py
# Builds FAISS index using ONLY first 100 rows (for development)

import json
import numpy as np
import faiss
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_PATH = "data/cardiology/raw/miriad_cardiology.json"
INDEX_PATH = "vectorstores/cardiology.faiss"
META_PATH = "vectorstores/cardiology_meta.json"
MAX_ROWS = 100  # <<< LIMIT HERE

device = "cuda" if torch.cuda.is_available() else "cpu"

# Models
instructor = SentenceTransformer("hkunlp/instructor-large").to(device)
proj = nn.Linear(768, 1024).to(device)

def build_index():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    # LIMIT TO FIRST 100 ROWS
    data = data[:MAX_ROWS]

    embeddings = []
    metadata = []

    for row in tqdm(data, desc="Building embeddings"):
        text = row["question"] + " " + row["answer"]

        emb_768 = instructor.encode([text])[0]
        emb_1024 = proj(
            torch.tensor(emb_768).float().unsqueeze(0).to(device)
        ).squeeze(0).detach().cpu().numpy()

        emb_1024 /= np.linalg.norm(emb_1024)

        embeddings.append(emb_1024)
        metadata.append(row)

    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(1024)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w") as f:
        json.dump(metadata, f)

    print(f"FAISS index built with {index.ntotal} documents")

if __name__ == "__main__":
    build_index()
