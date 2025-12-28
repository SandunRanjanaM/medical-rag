import json
import os
import faiss
import numpy as np
from haystack import Document, Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

from pathlib import Path

# project root = medical-rag/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = PROJECT_ROOT / "data/cardiology/raw/miriad_cardiology.json"
INDEX_PATH = PROJECT_ROOT / "backend/vectorstores/cardiology.faiss"
META_PATH = PROJECT_ROOT / "backend/vectorstores/cardiology_meta.json"

def build_index(limit=100):
    with open(DATA_PATH) as f:
        raw_data = json.load(f)[:limit]

    documents = []
    INSTRUCTION = "Represent the cardiology answer for retrieval:"

    for row in raw_data:
        if isinstance(row.get("answer"), str):
            documents.append(
                Document(
                    content=f"{INSTRUCTION}\n{row['answer']}",
                    meta=row
                )
            )

    document_store = InMemoryDocumentStore()

    indexing = Pipeline()
    indexing.add_component("cleaner", DocumentCleaner())
    indexing.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=8))
    indexing.add_component(
        "embedder",
        SentenceTransformersDocumentEmbedder(model="hkunlp/instructor-large")
    )
    indexing.add_component("writer", DocumentWriter(document_store))

    indexing.connect("cleaner", "splitter")
    indexing.connect("splitter", "embedder")
    indexing.connect("embedder", "writer")

    indexing.run({"cleaner": {"documents": documents}})

    return document_store
