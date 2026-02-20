import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ----------------------------
# Load model and tokenizer
# ----------------------------
@st.cache_resource(ttl=3600)  # reload every hour to get new model from HF
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

# ----------------------------
# App title and description
# ----------------------------
st.title("IMDB Sentiment Analyzer")
st.write("""
This app predicts whether an IMDB movie review is **Positive** or **Negative**  
using a **Transformer model (DistilBERT)** fine-tuned on 3,000 reviews.
""")

# ----------------------------
# User input
# ----------------------------
text = st.text_area("Enter a movie review here:", height=150)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter a review to predict!")
    else:
        with st.spinner("Predicting..."):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            )
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).squeeze()
                pred = torch.argmax(probs).item()
            
            sentiment = "Positive" if pred == 1 else "Negative"
            confidence = probs[pred].item() * 100

            # Display nicely
            st.subheader(f"Prediction: {sentiment}")
            st.progress(int(confidence))
            st.caption(f"Confidence: {confidence:.2f}%")
