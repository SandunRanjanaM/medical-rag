# frontend/streamlit_app/ui.py
import streamlit as st

def render_header():
    st.set_page_config(page_title="Medical RAG System", layout="centered")
    st.title("🩺 Multi-Specialty Medical RAG System")
    st.markdown(
        "Ask evidence-based medical questions across different specialties."
    )

def render_input_section():
    specialty = st.selectbox(
        "Select Medical Specialty",
        options=["cardiology", "neurology", "oncology", "orthopedics"]
    )

    question = st.text_area(
        "Enter your medical question",
        height=120,
        placeholder="e.g., What are the causes of myocardial infarction?"
    )

    submit = st.button("Submit Question")
    return specialty, question, submit

def render_answer(response: dict):
    if "error" in response:
        st.error(response["error"])
        return

    st.subheader("✅ Answer")
    st.write(response.get("answer", "No answer returned"))

    if "sources" in response:
        st.subheader("📚 Evidence")
        for src in response["sources"]:
            st.write(f"- {src}")
