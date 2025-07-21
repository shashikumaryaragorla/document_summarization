import streamlit as st
from transformers import pipeline

st.title("Text Summarizer App")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = st.text_area("Enter text to summarize:")

if st.button("Summarize"):
    if text:
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        st.subheader("Summary")
        st.write(summary[0]['summary_text'])
    else:
        st.warning("Please enter some text.")
