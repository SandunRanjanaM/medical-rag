# frontend/streamlit_app/app.py
import streamlit as st
from api import ask_backend
from ui import render_header, render_input_section, render_answer

def main():
    render_header()

    specialty, question, submit = render_input_section()

    if submit:
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Querying medical knowledge base..."):
                response = ask_backend(specialty, question)
            render_answer(response)

if __name__ == "__main__":
    main()
