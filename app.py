import os
import gradio as gr
# from dotenv import load_dotenv
from rag import rag_init, rag_query

# load_dotenv()

# FILE_PATH = os.getenv("RAG_FILE_PATH")
FILE_PATH = "harry_potter_knowledge_base.txt"
vector_store = rag_init(FILE_PATH)

def chat(message, history):
    return rag_query(vector_store, message)

gr.ChatInterface(fn=chat, title="RAG Chatbot").launch()