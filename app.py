import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

summarizer = load_model()

st.title("Text Summarization App")

text = st.text_area("Enter text to summarize")

if st.button("Summarize"):
    if text.strip():
        summary = summarizer(
            text,
            max_length=130,
            min_length=30,
            do_sample=False
        )
        st.write(summary[0]["summary_text"])
    else:
        st.warning("Please enter some text")

