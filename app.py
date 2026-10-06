import os

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI


load_dotenv()

DATA_DIR = "data"

Settings.llm = GoogleGenAI(model="gemini-3.8-flash")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

def get_query_engine():
	if not os.getenv("GEMINI_API_KEY"):
		st.error("GEMINI_API_KEY was not found. Check your .env file")
		st.stop()

	documents = SimpleDirectoryReader(DATA_DIR).load_data()
	index = VectorStoreIndex.from_documents(documents)

	return index.as_query_engine()

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
        query_engine = get_query_engine()
        response = query_engine.query(question)
        answer = response.response
        st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
