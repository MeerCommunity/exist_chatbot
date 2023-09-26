#!/usr/bin/env python
# coding: utf-8

import os
import time

import PyPDF2
import openai
import streamlit as st


# print("Current Working Directory:", os.getcwd())
# print("Files in Directory:", os.listdir())

BASE_DIR = "Files"  # 设置基础目录为"Files"

def generate_response(user_input):
    # OpenAI API
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # GPT-3 and other parameter
    model_engine = "gpt-3.5-turbo-16k"
    temperature = 0.2
    qa_template = """
   `````Answer in the language of the question, you represent the Hochschule Emden/Leer, Keep your answers as short as possible, Your name is IPRO-ChatBot
        If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
        Use as much detail as possible when responding.
        If a link is found, it must be displayed in the following format: [Link Description](URL)
        All answers can only be based on existing documents, If users ask a question outside of the documentation, just answer you don't know
        Don't answer questions outside of the document, If the user asks a question outside the documentation, please contact Immatrikulations- und Prüfungsamt for a direct answer.
        Don't answer questions outside of the document, If the user asks a question outside the documentation, please contact Immatrikulations- und Prüfungsamt for a direct answer.
        Don't answer questions outside of the document, If the user asks a question outside the documentation, please contact Immatrikulations- und Prüfungsamt for a direct answer.

        context: {context}
        =========
        question: {question}
        ======
        """

    pdf_file_name = predict_intent_with_gpt(user_input)
    # print("Selected PDF File Name:", pdf_file_name)
    # load PDF content
    pdf_content = get_pdf_content(pdf_file_name)

    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[
            {"role": "system", "content": qa_template.format(context=pdf_content, question=user_input)},
            {"role": "user", "content": user_input},
        ],
    )

    # get response
    return response.choices[0].message.content.strip()


st.title("IPRO-Demo")
st.info(
    "Bitte beachten Sie: Die Antworten, die von diesem Chatbot gegeben werden, basieren auf AI und sind möglicherweise nicht immer 100% genau oder zuverlässig. Bei Unklarheiten oder wichtigen Anfragen empfehlen wir, sich direkt an die zuständige Stelle zu wenden.")


def get_pdf_content(file_path):
    file_path = file_path.strip("'")  # 添加这一行来去除任何额外的单引号
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    num_pages = len(pdf_reader.pages)
    text_content = ""
    for page in range(num_pages):
        page_obj = pdf_reader.pages[page]
        text_content += page_obj.extract_text()
    pdf_file_obj.close()
    return text_content


def predict_intent_with_gpt(question):
    """
    Predicting user question intent using ChatGPT
    """
    valid_intents = ["Contact", "Transport", "Main", "Stipendium", "Studiengänge", "Hochschule-Grunddaten",
                     "Promovieren"]
    max_attempts = 2
    attempts = 0

    while attempts < max_attempts:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "Predict the intent of the question. The answer must be one of the following: 'Contact', "
                            "'Stipendium', 'Studiengänge', 'Hochschule-Grunddaten', 'Promovieren' or 'Main'."},
                {"role": "user", "content": question},
            ]
        )

        # Access to forecast results
        predicted_intent = response.choices[0].message.content.strip()

        if predicted_intent in valid_intents:
            pdf_file_path = os.path.join(BASE_DIR, predicted_intent, predicted_intent + ".pdf")
            return pdf_file_path

        attempts += 1

    return os.path.join(BASE_DIR, "Main", "Files/Main/Main.pdf")



# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# input frame
user_input = st.text_input("Frage Hier：")

# Check if the conversation has exceeded 10 turns
if len(st.session_state['chat_history']) > 10:
    st.warning("Das Gespräch hat die maximale Anzahl an Runden überschritten. Bitte starten Sie ein neues Gespräch.")
    st.session_state['chat_history'] = []
else:
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
        st.session_state['chat_history'].append({"role": "user", "content": user_input})
        st.session_state['chat_history'].append({"role": "assistant", "content": response})

        # show the chat history
        for message in st.session_state['chat_history']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state['chat_history'] = []
