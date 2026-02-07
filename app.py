import torch
import nltk
import streamlit as st

st.title("Torch + NLTK Debug")
st.write(f"torch: {torch.__version__}")
st.write(f"nltk: {nltk.__version__}")
