import os
import streamlit as st
import time
import io
import pandas as pd
import openai
import re
import PyPDF2
import numpy as np
import tiktoken
import time
from IPython.display import clear_output

openai.api_key = os.getenv("OPENAI_API_KEY")

start_time = time.time()

########### This helps takes care of removing metadata
search_string = "---"
metadata_counter = 0

# Extract text from PDF file
def extract_text_from_pdf(pdf_file):
    if isinstance(pdf_file, bytes):
        pdf_file = io.BytesIO(pdf_file)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    count = len(pdf_reader.pages)
    text = ""
    for i in range(count):
        page = pdf_reader.pages[i]
        text += page.extract_text()
    return text


# Normalizing text
def normalize_text(s, sep_token=" \n "):
    s = re.sub(r'\s+', ' ', s).strip()
    # ... rest of the function ...

# Adding token column to data
def add_token_column(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    if isinstance(text, bytes):  # if 'text' is a bytes object
        text = text.decode('utf-8')  # decode it to a string
    n_tokens = len(tokenizer.encode(text))
    return n_tokens

request_counter = 0
total_requests_sent = 0
rate_limit = 3000

start_timer = time.time()


def generate_embeddings(text, model="text-embedding-ada-002"):
    global request_counter
    global rate_limit
    global total_requests_sent
    global start_timer
    clear_output(wait=True)
    check_timer = time.time()
    duration = check_timer - start_timer
    print(duration)

    if int(duration) >= 60:
        start_timer = time.time()
        request_counter = 0
    if request_counter == rate_limit and int(duration) <= 59:
        sleep_for = 60 - int(duration)
        print("Sleeping for " + str(sleep_for) + " seconds")
        print("Total requests sent: ", total_requests_sent)
        time.sleep(sleep_for)
        start_timer = time.time()
        request_counter = 0
    if request_counter < rate_limit:
        request_counter += 1
        total_requests_sent += 1
        print("Request counter: ", request_counter)
        print("Total requests sent: ", total_requests_sent)

    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

# Streamlit app
def main():
    st.title('PDF Processing App')

    pdf_file = st.file_uploader("Upload PDF", type=['pdf'])

    if pdf_file is not None:
        pdf_text = extract_text_from_pdf(pdf_file)

        # Normalize text and add token column
        normalized_text = normalize_text(pdf_text)
        if not isinstance(normalized_text, str):
            normalized_text = str(normalized_text)
        n_tokens = add_token_column(normalized_text)

        # Convert data to dataframe
        df_tok = pd.DataFrame([{'CONTENT': normalized_text, 'n_tokens': n_tokens}])

        # Save dataframe to CSV file
        df_tok.to_csv('output.csv', index=False)
        st.write('CSV file saved.')

        # Generate and save embeddings
        df_embeddings = df_tok.copy()

        # Generate embeddings
        embeddings = generate_embeddings(df_tok.CONTENT[0], model='text-embedding-ada-002')

        # Convert the list of embeddings to a string
        embeddings_str = ', '.join(map(str, embeddings))

        # Assign the string of embeddings to the data frame
        df_embeddings['ada_v2_embedding'] = embeddings_str

        # Save dataframe to CSV file
        df_embeddings.to_csv('output.csv', index=False)

        # Create array of embeddings and save to Numpy file
        all_embeddings = np.array(df_embeddings['ada_v2_embedding'])
        np.save('output.npy', all_embeddings)
        st.write('Numpy file saved.')

if __name__ == "__main__":
    main()
