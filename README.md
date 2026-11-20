# Babson Handbook RAG Chatbot

OIM3641 Activities 10 and 11. Streamlit loads the handbook PDF, creates local
BAAI/bge-small-en-v1.5 embeddings and sends retrieved passages to Gemini.

```sh
uv sync --locked
uv run streamlit run app.py --server.address 127.0.0.1 --server.fileWatcherType none
```

Put GEMINI_API_KEY in the gitignored .env. The verified model is
`gemini-3.1-flash-lite`, with a 1,200-token output cap. The app caches its index
in memory; the first run needs model weights and later runs reuse the local
Hugging Face download. There is no need for Ollama, Whisper, or a new Jupyter
installation to run this assignment.

`Activity 10 RAG Chatbot.docx` contains the comparison and screenshots.
`evidence/rag-answers.json` records actual browser answers; the general chat
comparison used Gemini without the handbook. The visible message history is
not sent to the query engine, so this app does not have conversational memory.
The embedded handbook is a course-provided snapshot, not a guarantee of current
college policy. Code, verification and this completion used ChatGPT assistance.
