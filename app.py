import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

summarizer = load_model()

def chunk_text(text, max_words=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks

st.title("📄 Document Summarization App")

text = st.text_area("Enter text to summarize", height=250)

if st.button("Summarize"):
    if not text.strip():
        st.warning("Please enter some text")
    else:
        chunks = chunk_text(text)

        summaries = []
        for chunk in chunks:
            result = summarizer(
                chunk,
                max_length=130,
                min_length=30,
                do_sample=False
            )
            summaries.append(result[0]["summary_text"])

        final_summary = " ".join(summaries)

        st.subheader("Summary")
        st.write(final_summary)
