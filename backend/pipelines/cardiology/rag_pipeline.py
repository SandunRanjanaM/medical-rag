import numpy as np
from typing import List
from haystack import Pipeline, Document, component
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from .models import l2_normalize

# --------------------
# HyDE Components
# --------------------
@component
class HypothesisConverter:
    def __init__(self, instruction: str):
        self.instruction = instruction

    @component.output_types(documents=List[Document])
    def run(self, replies: List[str]):
        return {
            "documents": [
                Document(content=f"{self.instruction}\n{reply}")
                for reply in replies
            ]
        }

@component
class HypotheticalDocumentEmbedder:
    def __init__(
        self,
        instruct_llm="SandunR/hyde-sciFive-cardiology-generator",
        embedder_model="hkunlp/instructor-large",
        nr_completions=4,
    ):
        self.INSTRUCTION = "Represent the cardiology document for retrieval:"

        self.generator = HuggingFaceLocalGenerator(
            model=instruct_llm,
            task="text2text-generation",
            generation_kwargs={
                "do_sample": True,
                "num_beams": 4,
                "num_return_sequences": nr_completions,
                "temperature": 0.7,
                "max_new_tokens": 300,
            },
        )

        self.prompt_builder = PromptBuilder(
            template="Question: {{question}}\nParagraph:",
            required_variables=["question"],
        )

        self.converter = HypothesisConverter(self.INSTRUCTION)

        self.embedder = SentenceTransformersDocumentEmbedder(
            model=embedder_model,
            progress_bar=False,
        )
        self.embedder.warm_up()

        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt", self.prompt_builder)
        self.pipeline.add_component("generator", self.generator)
        self.pipeline.add_component("converter", self.converter)
        self.pipeline.add_component("embedder", self.embedder)

        self.pipeline.connect("prompt", "generator")
        self.pipeline.connect("generator.replies", "converter.replies")
        self.pipeline.connect("converter.documents", "embedder.documents")

    @component.output_types(hypothetical_embedding=List[float])
    def run(self, query: str):
        result = self.pipeline.run({"prompt": {"question": query}})
        docs = result["embedder"]["documents"]
        stacked = np.array([doc.embedding for doc in docs])
        avg = stacked.mean(axis=0)
        return {"hypothetical_embedding": l2_normalize(avg).tolist()}

# --------------------
# Fusion
# --------------------
def fuse_embeddings(proj_query, hyde_query, alpha=0.5):
    fused = alpha * proj_query + (1 - alpha) * hyde_query
    return l2_normalize(fused)
