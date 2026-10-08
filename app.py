import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
import streamlit as st


load_dotenv()

DATA_DIR = Path("data")


def get_api_key():
    """Return the Gemini API key or stop with a helpful message."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error(
            "GEMINI_API_KEY is missing. Add it to your .env file as "
            "GEMINI_API_KEY=your-key, then restart the app."
        )
        st.stop()

    return api_key


def validate_data_directory():
    """Check that the handbook data folder is ready."""
    if not DATA_DIR.is_dir():
        st.error(
            f"Data folder not found: {DATA_DIR}. "
            "Create that folder and put the handbook PDF inside it."
        )
        st.stop()

    files = [
        path for path in DATA_DIR.iterdir()
        if path.is_file() and not path.name.startswith(".")
    ]

    if not files:
        st.error(
            f"The {DATA_DIR} folder is empty. "
            "Add the handbook PDF, then restart the app."
        )
        st.stop()


@st.cache_resource
def get_query_engine(api_key):
    """Build and return the handbook query engine."""
    Settings.llm = GoogleGenAI(
        model="gemini-3.8-flash",
        api_key=api_key
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()
    index = VectorStoreIndex.from_documents(documents)

    return index.as_query_engine()


api_key = get_api_key()
validate_data_directory()

try:
    query_engine = get_query_engine(api_key)
except Exception as error:
    st.error(
        "Could not build the handbook search. Check that the PDF is valid "
        f"and restart the app. Details: {error}"
    )
    st.stop()


st.title("Babson Student Handbook Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask a question about the handbook")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            response = query_engine.query(question)
            answer = response.response
            st.write(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        except Exception as error:
            st.error(
                "Could not answer that question. Check your internet "
                f"connection and try again. Details: {error}"
            )
