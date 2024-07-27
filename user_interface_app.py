import numpy as np
import pickle
import streamlit as st
import os
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import time
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(premise,hypothesis):
    return tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding='max_length', max_length=128)

def predict(premise, hypothesis):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loaded_model.to(device)
    inputs = tokenize(premise, hypothesis)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = loaded_model(**inputs)

    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=-1).item()

    label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    return label_map[predicted_label]

loaded_model=BertForSequenceClassification.from_pretrained(r'C:\Users\HP\Desktop\IITI_SoC')
#loaded_model = pickle.load(open("C:/Users/Vaibhav/Desktop/machine learning/trained_model.sav",'rb'))

def sentence_prediction(premise_, hypothesis_):
    prediction= predict(premise_,hypothesis_)
    return prediction
    
def main():
    st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://blog.rapid7.com/content/images/2023/12/GettyImages-1470617656.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
     # App title and headings
    col1, col2, col3 = st.columns([2, 2, 1])
    col1.markdown("# Welcome to our app!")
    st.title("SENTENCE RELATION IDENTIFIER")

    # Input fields
    premise_ = st.text_input("Enter the first sentence")
    hypothesis_ = st.text_input("Enter the second sentence")

    # code for prediction
    result = ''
    # button creation
    if st.button('Check the relation'):
        progress_bar = st.progress(0)
        for i in range(100):   
            time.sleep(0.002)
            progress_bar.progress(i + 1)
        result= sentence_prediction(premise_, hypothesis_)
       
    st.success(result)


if __name__ == '__main__':
    main()
