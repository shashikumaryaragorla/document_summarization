import streamlit as st
from transformers import pipeline
import torch

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1  # Force CPU (saves memory)
    )

summarizer = load_model()

def chunk_text(text, max_words=400):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

st.title("📄 Document Summarization App")

text = st.text_area("Enter text", height=250)

if st.button("Summarize"):
    if text.strip():
        summaries = []
        for chunk in chunk_text(text):
            out = summarizer(
                chunk,
                max_length=120,
                min_length=30,
                do_sample=False
            )
            summaries.append(out[0]["summary_text"])

        st.subheader("Summary")
        st.write(" ".join(summaries))
    else:
        st.warning("Please enter text")

