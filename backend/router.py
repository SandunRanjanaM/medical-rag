from pipelines.cardiology.rag_pipeline import cardiology_qa

def route_question(specialty, question):
    if specialty == "cardiology":
        return cardiology_qa(question)
    else:
        return {"error": "Invalid specialty"}
