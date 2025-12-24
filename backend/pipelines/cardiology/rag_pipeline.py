# backend/pipelines/cardiology/rag_pipeline.py
import json
import faiss
import numpy as np
import torch

from .models import (
    instructor, inbedder, proj,
    hyde_model, hyde_tokenizer,
    final_model, final_tokenizer, device
)

INDEX_PATH = "vectorstores/cardiology.faiss"
META_PATH = "vectorstores/cardiology_meta.json"

index = faiss.read_index(INDEX_PATH)
with open(META_PATH) as f:
    kb_text = json.load(f)

def generate_hyde(query):
    inputs = hyde_tokenizer(
        f"hypothetical answer: {query}",
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    outputs = hyde_model.generate(**inputs, max_length=180)
    return hyde_tokenizer.decode(outputs[0], skip_special_tokens=True)

def cardiology_qa(query, alpha=0.5, top_k=5):
    # HyDE embedding
    hyde_doc = generate_hyde(query)
    hyde_emb = instructor.encode([hyde_doc])[0]
    hyde_emb = proj(torch.tensor(hyde_emb).float().unsqueeze(0).to(device))
    hyde_emb = hyde_emb.squeeze(0).detach().cpu().numpy()
    hyde_emb /= np.linalg.norm(hyde_emb)

    # Query embedding
    q_emb = inbedder.encode([query], "medical query: ")[0]
    q_emb /= np.linalg.norm(q_emb)

    # Combine
    final_emb = alpha * q_emb + (1 - alpha) * hyde_emb
    final_emb /= np.linalg.norm(final_emb)

    # Retrieve
    _, I = index.search(final_emb.reshape(1, -1).astype("float32"), top_k)
    retrieved = [kb_text[i] for i in I[0]]

    # Generate answer
    context = "\n".join([r["answer"] for r in retrieved])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    inputs = final_tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = final_model.generate(**inputs, max_length=256)

    answer = final_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "answer": answer,
        "hyde": hyde_doc,
        "sources": retrieved
    }
