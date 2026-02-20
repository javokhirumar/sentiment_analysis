# IMDB Sentiment Analysis

We compared a custom RNN and a Transformer (DistilBERT) for IMDB sentiment analysis.  
- The RNN achieved **50.6% accuracy** with a **loss of 0.696**.  
- The Transformer had a lower **loss of 0.536**, showing that Transformers outperform RNNs for NLP tasks, especially on limited data.  

## Try the live app

You can test the model with this Streamlit app: [Sentiment Prediction](https://sentiment-demo.streamlit.app/)

## Repo contents

- `sentiment_analysis.ipynb` — Notebook with RNN and Transformer experiments  
- `app.py` — Streamlit app for real-time predictions  
- `requirements.txt` — Python dependencies

## Notes

- The Streamlit app downloads the Hugging Face model automatically.  
- No large model files are stored in this GitHub repo.
