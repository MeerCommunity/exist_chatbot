#!/usr/bin/env python
# coding: utf-8

import os
import streamlit as st
import time
import openai

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

# input frame
user_input = st.text_input("Frage Hier：")

if st.button("Send"):
    # with a waiting icon
    # GPT needs some time to response
    with st.spinner("Waiting..."):
        time.sleep(2)

    # get generate_response
    response = generate_response(user_input)

    # show the anwser
    st.write(response)
