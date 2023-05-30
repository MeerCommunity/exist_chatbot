#!/usr/bin/env python
# coding: utf-8

import os
import streamlit as st
import time
import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import cosine_similarity
from streamlit_chat import message

def generate_response(user_input):
    # OpenAI API
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # GPT-3 and other parameter
    model_engine = "gpt-3.5-turbo"

    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[
            {"role": "system", "content": "Antwort auf Deutsch, Sie vertreten die Hochschule Emden/Leer, Halten Sie Ihre Antworten so kurz wie möglich, Ihr Name ist IPRO-ChatBot"},
            {"role": "user", "content": user_input},
        ],
    )

    # get response
    return response.choices[0].message.content.strip()

st.title("IPRO-Demo")

def get_embedding(text, model="text-embedding-ada-002"):
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def search_docs(df, user_query, top_n=3):
    embedding = get_embedding(user_query, model="text-embedding-ada-002")
    df_question = df.copy()
    df_question["similarities"] = df.ada_v2_embedding.apply(lambda x: cosine_similarity(x, embedding))
    res = df_question.sort_values("similarities", ascending=False).head(top_n)
    return res

# Load the dataframe and embeddings
df_try = pd.read_csv('output.csv')
all_embeddings = np.load('output.npy', allow_pickle=True)
df_try['ada_v2_embedding'] = all_embeddings


# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# input frame
user_input = st.text_input("Frage Hier：")

if st.button("Send"):
    # with a waiting icon
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # GPT needs some time to response
    with st.spinner("Waiting..."):
        time.sleep(2)

    # get generate_response
    response = generate_response(user_input)

    # Add user input and response to chat history
    st.session_state['chat_history'].append((user_input, response))

    # show the chat history
    for user_msg, bot_msg in st.session_state['chat_history']:
        message(user_msg, is_user=True)
        message(bot_msg)

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state['chat_history'] = []
