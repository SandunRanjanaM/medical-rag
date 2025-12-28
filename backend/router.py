from pathlib import Path
import torch
from fastapi import APIRouter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from pipelines.cardiology.build_index import build_index
from pipelines.cardiology.models import (
    embed_user_query,
    project_user_query,
    ProjectionLayer
)
from pipelines.cardiology.rag_pipeline import (
    HypotheticalDocumentEmbedder,
    fuse_embeddings
)

BASE_DIR = Path(__file__).resolve().parents[1]  
# backend/ → medical-rag/

PROJECTOR_PATH = BASE_DIR / "models" / "cardiology" / "projector.pt"
router = APIRouter()

# Initialize once
document_store = build_index()
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
hyde = HypotheticalDocumentEmbedder()

projector = ProjectionLayer()
projector.load_state_dict(torch.load(PROJECTOR_PATH, map_location="cpu"))
projector.eval()

@router.post("/ask/cardiology")
def ask_cardiology(question: str):
    inbed_emb = embed_user_query(question)
    proj_emb = project_user_query(projector, inbed_emb)
    hyde_emb = hyde.run(question)["hypothetical_embedding"]

    final_emb = fuse_embeddings(proj_emb, hyde_emb)

    results = retriever.run(
        query_embedding=final_emb.tolist(),
        top_k=5
    )

    return {
        "answers": [doc.content for doc in results["documents"]]
    }


# Add a /ask route for compatibility
from fastapi import Request
@router.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question")
    # Optionally handle specialty, but for now just use cardiology logic
    return ask_cardiology(question)
