# backend/pipelines/cardiology/models.py
import torch
import torch.nn as nn
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    AutoModelForMaskedLM, AutoTokenizer
)
from sentence_transformers import SentenceTransformer
from torch.nn.functional import gelu

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instructor
instructor = SentenceTransformer("hkunlp/instructor-large").to(device)

# Projection
proj = nn.Linear(768, 1024).to(device)

# InBedder
class InBedder:
    def __init__(self, path="KomeijiForce/inbedder-roberta-large"):
        model = AutoModelForMaskedLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = model.roberta.to(device)
        self.dense = model.lm_head.dense.to(device)
        self.layer_norm = model.lm_head.layer_norm.to(device)

    def encode(self, texts, instruction, n_mask=3):
        prompts = [instruction + self.tokenizer.mask_token * n_mask] * len(texts)
        inputs = self.tokenizer(
            texts, prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        mask = inputs.input_ids.eq(self.tokenizer.mask_token_id)
        outputs = self.model(**inputs)
        logits = outputs.last_hidden_state[mask]
        logits = self.layer_norm(gelu(self.dense(logits)))
        logits = logits.view(len(texts), n_mask, -1).mean(1)
        return logits.detach().cpu().numpy()

inbedder = InBedder()

# HyDE model
hyde_name = "SandunR/hyde-sciFive-cardiology-generator"
hyde_tokenizer = T5Tokenizer.from_pretrained(hyde_name)
hyde_model = T5ForConditionalGeneration.from_pretrained(hyde_name).to(device)

# Final generator
final_name = "google/flan-t5-large"
final_tokenizer = T5Tokenizer.from_pretrained(final_name)
final_model = T5ForConditionalGeneration.from_pretrained(final_name).to(device)
