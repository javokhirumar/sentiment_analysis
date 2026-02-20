import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Cache model so it loads only once
@st.cache_resource(ttl=3600)
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        "javokhirumar/sentiment-analysis"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "javokhirumar/sentiment-analysis"
    )
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

st.title("IMDB Sentiment Analyzer")

text = st.text_area("Enter a review:")

if st.button("Predict"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=1).item()

    st.write("Prediction:", "Positive" if pred == 1 else "Negative")
