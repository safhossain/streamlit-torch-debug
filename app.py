import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn import functional as F

import re
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

import streamlit as st



class FinancialSentimentDataset(Dataset):
    def __init__(self, vocab=None, max_length=50):
        self.lemmatizer = WordNetLemmatizer()
        self.vocab = vocab
        self.max_length = max_length
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    def preprocess_text(self, text):
        text = text.lower()
        # keep important numbers
        text = re.sub(r'\$\d+(?:\.\d+)?', '<CURRENCY>', text)
        text = re.sub(r'\d+(?:\.\d+)?%', '<PERCENT>', text)
        text = re.sub(r'\b\d+(?:\.\d+)?\b', '<NUMBER>', text)
        # keep tickers (acronyms)
        text = re.sub(r'\b[A-Z]{1,4}(?:\.[A-Z])?\b(?=\s|$)', '<TICKER>', text)
        # keep negations
        text = text.replace("n't", " not").replace("'t", " not")
        # avoid lemmatizing idioms
        tokens = text.split()
        financial_terms = {'bull', 'bear', 'short', 'long', 'rally', 'plunge'}
        tokens = [self.lemmatizer.lemmatize(t) if t not in financial_terms else t for t in tokens]
        return tokens
    
    def text_to_sequence(self, text):
        tokens = self.preprocess_text(text)
        sequence = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        if len(sequence) < self.max_length:
            sequence = sequence + [self.vocab['<PAD>']] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
            
        return sequence

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, n_layers=2, output_dim=3, dropout=0.5):
        super(SentimentLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers,
            batch_first=True, 
            dropout=dropout,
            bidirectional=True
        )
        
        # assign more weight to specific words in sequence
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)

        #self.fc = nn.Linear(hidden_dim * 2, output_dim)
        # features -> output
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        aweights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(aweights * lstm_out, dim=1)
        
        hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden_concat = hidden_concat + context
        
        # normalization and dropout
        hidden_concat = self.batch_norm(hidden_concat)
        hidden_concat = self.dropout(hidden_concat)
        
        output = self.classifier(hidden_concat)
        return output


@st.cache_resource
def load_model():
    # load model
    checkpoint = torch.load('best_tuned_model.pth', map_location=torch.device('cpu'))
    
    dataset = FinancialSentimentDataset(
        vocab=checkpoint['vocab'],
        max_length=checkpoint['max_length']
    )
    
    # make model from cfg
    config = checkpoint.get('model_config', {})
    model = SentimentLSTM(
        vocab_size=len(checkpoint['vocab']),
        embedding_dim=config.get('embedding_dim', 192),
        hidden_dim=config.get('hidden_dim', 128),
        n_layers=config.get('n_layers', 3),
        output_dim=3,
        dropout=config.get('dropout', 0.5)
    )
    
    # trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
        
    return model, dataset, checkpoint

st.title("Financial Headlines")
st.subheader("Sentiment Analysis")
st.write("Classify Headline as Negative, Neutral, or Positive")

model, dataset, checkpoint = load_model()

# input
text_input = st.text_area(
    "Type Headline...",
    height=100
)

if st.button("Analyze Sentiment"):
    # preprocess and predict
    sequence = torch.tensor([dataset.text_to_sequence(text_input)], dtype=torch.long)
            
    with torch.no_grad():
        output = model(sequence)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
    sentiment = sentiment_labels[prediction]
    
    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sentiment", sentiment)
    with col2:
        st.metric("Confidence", f"{confidence:.2%}")
            
    # get probabilities
    st.subheader("Probability Distribution")
    labels = ['negative', 'neutral', 'positive']
    probs = probabilities[0].tolist()
    for label, prob in zip(labels, probs):
        st.progress(prob, text=f"{label.title()}: {prob:.2%}")
            
    # model
    with st.expander("Model Info"):
        st.write(f"**Training Accuracy:** {checkpoint.get('train_accuracy', 'N/A')/100:.2%}")
        st.write(f"**Validation Accuracy:** {checkpoint.get('val_accuracy', 'N/A')/100:.2%}")
        st.write(f"**Epochs Trained:** {checkpoint.get('epoch', 'N/A')}")
        st.write(f"**Vocabulary Size:** {len(checkpoint['vocab'])}")
