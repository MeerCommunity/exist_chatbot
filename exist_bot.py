import os
import time
import PyPDF2
import openai
import streamlit as st

BASE_DIR = "Files"  # 设置基础目录为"Files"

def generate_response(user_input):
    # OpenAI API
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # GPT-3 and other parameter
    model_engine = "gpt-3.5-turbo-16k"
    temperature = 0.2
    qa_template = """
    Answer in the language of the question. If you're unsure or don't know the answer, respond with "Ich weiß es nicht,
    bitte wenden Sie sich an die zuständige Abteilung der HSEL".
    You represent the Hochschule Emden/Leer and your name is IPRO-ChatBot.
    Only answer based on the provided context. If the question is outside of the context, say "I don't know".
    For example:
        question: "What's the capital of France?"
        answer: "I don't know"

    context: {context}
    ========
    question: {question}
    ======
    """
    pdf_file_name = predict_intent_with_gpt(user_input)
    pdf_content = get_pdf_content(pdf_file_name)
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[
            {"role": "system", "content": qa_template.format(context=pdf_content, question=user_input)},
            {"role": "user", "content": user_input},
        ],
    )
    return response.choices[0].message.content.strip()

def get_pdf_content(file_path):
    file_path = file_path.strip("'")
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
    valid_intents = ["Contact", "Transport", "Main", "Stipendium", "Studiengänge", "Hochschule-Grunddaten", "Promovieren", "Person"]
    max_attempts = 2
    attempts = 0
    while attempts < max_attempts:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "Predict the intent of the question. The answer must be one of the following: 'Contact', 'Stipendium', 'Studiengänge', 'Hochschule-Grunddaten', 'Promovieren' , 'Person' or 'Main'."},
                {"role": "user", "content": question},
            ]
        )
        predicted_intent = response.choices[0].message.content.strip()
        if predicted_intent in valid_intents:
            pdf_file_path = os.path.join(BASE_DIR, predicted_intent, predicted_intent + ".pdf")
            return pdf_file_path
        attempts += 1
    return os.path.join(BASE_DIR, "Main", "Files/Main/Main.pdf")

st.title("IPRO-Demo")
st.info("Bitte beachten Sie: Die Antworten, die von diesem Chatbot gegeben werden, basieren auf AI und sind möglicherweise nicht immer 100% genau oder zuverlässig. Bei Unklarheiten oder wichtigen Anfragen empfehlen wir, sich direkt an die zuständige Stelle zu wenden.")

# Initialize chat history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
user_input = st.chat_input("Frage Hier：")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    response = generate_response(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
