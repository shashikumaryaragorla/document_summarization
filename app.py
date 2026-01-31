import streamlit as st
import torch
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1,
        torch_dtype=torch.float32
    )

summarizer = load_model()

def chunk_text(text, max_words=350):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

st.title("Text Summarization App")

text = st.text_area("Paste text (max 4000 chars)", height=220)
text = text[:4000]

if st.button("Summarize"):
    with st.spinner("Summarizing..."):
        summaries = []
        for chunk in chunk_text(text):
            result = summarizer(
                chunk,
                max_length=100,
                min_length=30,
                do_sample=False
            )
            summaries.append(result[0]["summary_text"])

        st.subheader("Summary")
        st.write(" ".join(summaries))
