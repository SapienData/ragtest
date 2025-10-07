import streamlit as st
from mistralai import Mistral
import numpy as np
import faiss
import pandas as pd
from docx import Document
import time
import os


api_key = st.secrets["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)


file_path = "source.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()


chunk_size = 4096
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def get_text_embedding(input):
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=input
    )
    return embeddings_batch_response.data[0].embedding


@st.cache_data
def embed_chunks(chunks):
    import tqdm
    embeddings = []
    for chunk in tqdm.tqdm(chunks, desc="Embedding chunks"):
        emb = get_text_embedding(chunk)
        embeddings.append(emb)
        time.sleep(1.05)
    return np.array(embeddings)

text_embeddings = embed_chunks(chunks)

d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)


st.title("Local RAG Q&A Interface")

user_question = st.text_input("Ask a question:")

if user_question:
    question_embeddings = np.array([get_text_embedding(user_question)])

    D, I = index.search(question_embeddings, k=2)
    retrieved_chunks = [chunks[i] for i in I.tolist()[0]]

    prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunks}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {user_question}
    Answer:
    """

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.complete(model="mistral-large-latest", messages=messages)
    answer = response.choices[0].message.content

    st.markdown("Answer:")
    st.write(answer)
