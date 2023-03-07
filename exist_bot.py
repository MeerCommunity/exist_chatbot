#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import io
import time
import pandas as pd
import openai
import numpy as np
import streamlit as st
import requests
import webbrowser
from openai.embeddings_utils import cosine_similarity
from num2words import num2words
from streamlit.components.v1 import html


openai.api_key = st.secrets["api_key"]


start_time=time.time()

########### This helps takes care of removing metadata
search_string = "---" 
metadata_counter = 0
############




messages = [
    {"role": "system", "content": ""},
]

#Erstelle einen Datframe mit Inhalten und den dazugehörigen Embeddings
df_try =pd.read_csv('df_chatbot_exist_v4.csv')
all_embeddings = np.load('embeddings_v4.npy', allow_pickle=True)
df_try['ada_v2_embedding'] = all_embeddings


# Set page config
st.set_page_config(page_title="EXIST-Chatbot", page_icon="MCSC_Icon.png", layout="wide")

def clear_input():
    message.empty()

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.meercommunity.de/wp-content/uploads/2022/09/Hover_bg.png");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

button_style = """
    <style>
        .stButton > button{
            color: #211f39;
            height: 100px;
            width: 200px;
            background: #0069af;
            }
         .submitBtn > button {
            color: white;
            height: 50px;
            width: 100px;
            background: red;
            }
            

     </style>
      """
st.markdown(button_style, unsafe_allow_html=True)

button_style_submit = """
    <style>
         .submitBtn{
            color: white;
            height: 50px;
            width: 100px;
            background: red;
            }
     </style>
      """
button_styles_try = {
    "background-color": "#007bff",
    "color": "#fff",
    "font-size": "20px",
    "padding": "10px 20px",
    "border-radius": "5px",
}

def get_embedding(text, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def open_URL(url):
    webbrowser.open(url)

def update_markdown(empty_slot, new_text):
    empty_slot.markdown(f"""
        <style>
            .my-container {{
                background-color: #444076;
                color: white;
                font-size: 24px;
            }}
        </style>
        <div class="my-container">
            {new_text}
        </div>
    """, unsafe_allow_html=True)

def search_docs(df, user_query, top_n=3, to_print=True):
    embedding = get_embedding(
        user_query,
        model="text-embedding-ada-002"
    )
    #Erstelle eine Kopie des Datframes  
    df_question = df.copy()
    
    #Füge eine weitere Spalte hinzu mit Similarity-Score
    df_question["similarities"] = df.ada_v2_embedding.apply(lambda x: cosine_similarity(x, embedding))
    
    #Sortiere den Dataframe basierend auf dem Similarity-Score und zeige die obersten N Einträge
    res = (
        df_question.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    #Greife die obersten N Einträge ab
    return res

Willkommenstext = "Willkommen! Ich bin ein Chatbot für das EXIST-Gründerstipendium, dein persönlicher Assistent, der dir bei Fragen rund um das Gründerstipendium hilft. Egal, ob du Informationen zu den Anforderungen, dem Bewerbungsprozess oder anderen Themen benötigst, ich stehe dir gerne zur Seite. Stelle einfach deine Fragen und ich werde mein Bestes geben, um dir weiterzuhelfen."
ai_response = ""
ai_question = ""
MCSC_url = "https://www.meercommunity.de"
Wer_url ="https://www.meercommunity.de/meercommunity/?_gl=1*fbcg3r*_ga*MTEzMTcxMDQyOS4xNjc4MDkyNzc5*_up*MQ.."
Beratung_url = "https://www.meercommunity.de/booking/?_gl=1*v41p4s*_ga*MTEzMTcxMDQyOS4xNjc4MDkyNzc5*_up*MQ.."
Veranstaltung_url ="https://www.meercommunity.de/veranstaltung/?_gl=1*fbcg3r*_ga*MTEzMTcxMDQyOS4xNjc4MDkyNzc5*_up*MQ.."


   
if __name__== '__main__':
    #while True:
    # Add custom CSS for buttons
    st.markdown("""
    <style>
        .burger {
            display: none;
            color: white !important;
                }
        .navbar {
            overflow: hidden;
            background-color: #211f39;
            display: flex;
            justify-content: center;
            align-items: top; /* Zentriert die Elemente vertikal */
            border-bottom: 1px solid white;
                }

        .navbar a {
            float: left;
            display: block;
            color: white;
            margin-right: 30px; /* Erhöht den Abstand zwischen den Elementen */
            font-size: 30px; /* Ändert die Schriftgröße */
            text-align: center;
            text-decoration: none;
            padding-top:14px; 
            padding-bottom:14px; 
            padding-left:16px; 
            padding-right:16px; 
                }

        .navbar a:hover {
            border-bottom: 3px solid white;
            }
            

        .navbar img {
            height: 80px; /* Setzt die Höhe des Bildes auf auto */
            width:80px; /* Setzt die Breite des Bildes auf auto */
            max-height:80px; /* Setzt die maximale Höhe des Bildes auf auto */
            max-width:80px; /* Setzt die maximale Breite des Bildes auf auto */
            vertical-align:center; /* Zentriert das Bild vertikal */
            }
            
     @media screen and (max-width: 600px) {
            /* Zeigt das Burger-Icon auf kleinen Bildschirmen an */
        .burger {
            display: block;
            justify-content: center;
            color: white !important;
            font-size: 30px; /* Ändert die Schriftgröße */
            text-align: center;
            text-decoration: none;
            padding-top:14px; 
            padding-bottom:14px; 
            padding-right:16px; 
    }

    /* Versteckt die Navbar-Elemente auf kleinen Bildschirmen */
        .navbar a {
        display: none;
    }

    /* Zeigt die Navbar-Elemente an, wenn das Menü geöffnet ist */
    .navbar.open a {
        display: block;
        color: white !important;
    }
        }
    </style>
    
    <!-- Das Burger-Menü -->
    <div class="burger"> 
        <div> <a href="https://www.meercommunity.de/meercommunity/">ÜBER UNS</a> </div>
        <div><a href="https://www.meercommunity.de/veranstaltung/">VERANSTALTUNGEN</a> </div>
        <div><a href="https://www.meercommunity.de/informieren/">INFORMIEREN</a></div>
        <div> <a href="https://www.meercommunity.de/gruenderinnen/">BERATUNG</a> </div>
        <div><a href="https://www.meercommunity.de/gruenderinnen/">BERATUNG</a> </div>
        <div><a href="https://www.meercommunity.de/news/">NEWS</a></div>
    </div>

    <div class="navbar">
        <a href="https://www.meercommunity.de/"><img src="https://www.meercommunity.de/wp-content/uploads/2023/03/MCSC_Logo.png"></a>
        <a href="https://www.meercommunity.de/meercommunity/">ÜBER UNS</a>
        <a href="https://www.meercommunity.de/veranstaltung/">VERANSTALTUNGEN</a>
        <a href="https://www.meercommunity.de/informieren/">INFORMIEREN</a>
        <a href="https://www.meercommunity.de/netzwerken/">NETZWERK</a>
        <a href="https://www.meercommunity.de/gruenderinnen/">BERATUNG</a>
        <a href="https://www.meercommunity.de/news/">NEWS</a>
    </div>
    """, unsafe_allow_html=True)
    
    # Fügt das JavaScript hinzu
    html("""
    <script>
       // Wählt das Burger-Icon und die Navbar aus
        const burger = document.querySelector('.burger');
        const navbar = document.querySelector('.navbar');

       // Fügt einen Event-Listener zum Burger-Icon hinzu
        burger.addEventListener('click', () => {
    // Schaltet die Klasse "open" für die Navbar um
        navbar.classList.toggle('open');
        });
    </script>
""")
    keyInt = 0
    #input = ""
    # Render page layout
    
    Beispiel1 = "Was ist Exist?              "             
    Beispiel2 = "Was macht die Meercommunity?"
    Beispiel3 = "Welche Förderungen gibt es?"
    
    initial_prompt ="Du bist ein Start-Up Guide für das EXIST Programm. Du bist hilfreich, clever und freundlich. Antworte ausschließlich in deutscher Sprache. Sei prägnant in deinen Antworten. Nutze den folgenden Text für deine Antwort:"
    #ai_question = st.text_input('Wie kann ich dir helfen???', key=str(keyInt))
            # Define colors
    bg_color = "#211f39" # marine blue
    text_color = "#ffffff" # white
    
    # Define logos
    logo1 = "MCSC_Logo.png"
    logo2 = "Koop_Innosys_NW.png"
    logoBMWK ="logo_BMWK.png"
    logoHSEL ="logo_hsel.png"
    logoEXIST ="logo_exist.png"
    notice = "hinweis_beta.png"
    termin = "Termin_vereinbaren_rund.png"
    
    # Set page background color
    page_bg = f"background-color: {bg_color};"
    st.markdown("<style>body{background-color: Blue;}</style>",unsafe_allow_html=True)
    
    # Create header section
    header_bg = f"background-color: {bg_color}; border-radius: 5px; padding: 10px;"
    st.markdown(f"<div style='{header_bg}'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2,1])
    with col1:
        st.markdown(f'<a href="https://www.meercommunity.de/booking/?_gl=1*3tmwvy*_ga*MjU0NTUwNjgxLjE2NzgxMTQ3NzU.*_up*MQ.."><img src="https://www.meercommunity.de/wp-content/uploads/2022/07/Termin_vereinbaren_rund.png" style="width: 300px" style="justify-content: center;"></a>', unsafe_allow_html=True)
    with col2:
        
        st.markdown(f'<p style="color:{text_color};font-size:72px;border-radius:2%;text-align:center;">EXIST-Chatbot</p>', unsafe_allow_html=True)
                    # Create input field section
        input_bg = f"background-color: {bg_color}; border-radius: 5px; padding: 10px; margin-top: 20px;"
        st.markdown(f"<div style='{input_bg}'>", unsafe_allow_html=True)
        st.write("")
        #st.markdown(f'<p style="color:{text_color};font-size:24px;border-radius:2%;">"Schön, dass ihr an den Themen Exist und Gründung interessiert seid. Ich habe alle Antworten zum Thema EXIST-Stipendium und Antragsstellung. Naja, fast alle. Meine menschlichen Sklaven, äh, Helfer kann ich dann doch noch nicht entbehren. Sagen die zumindest. Naja, findet es selber heraus, und gebt eure Fragen in das Eingabefeld ein."</p>', unsafe_allow_html=True)
        #message = st.text_input("")
        st.markdown(f'<p style="color:{text_color};font-size:24px;border-radius:2%; font-style:italic;"> {Willkommenstext}</p>', unsafe_allow_html=True)
        st.write("")
        q0, q1, q2, q3 = st.columns([1,1, 1,1])
        with q0:
            st.write("")
            st.write("")
            #st.write("Beispielfragen:")
            st.markdown(f'<p style="color:{text_color};font-size:24px;border-radius:2%;text-align:right;vertical-align:middle;"> Beispielfragen:</p>', unsafe_allow_html=True)
        with q1:
            q1_btn = st.button(Beispiel1, "q1")
        with q2:
            q2_btn = st.button(Beispiel2, "q2")
        with q3:
            q3_btn = st.button(Beispiel3, "q3")
        st.write("")
        st.write("")
        st.markdown('<style>label {color: white;}</style>', unsafe_allow_html=True)
        message = st.text_input(label="Bitte gib eine Nachricht ein:",key= "input", value="Wer bist du?")
       
        
        #message = st.text_area("", "")
        st.markdown("</div>", unsafe_allow_html=True)
  
        #submit = st.button("Abschicken", "")
        button = st.button("Abschicken", key="my_button", help="Klicke mich um die Frage abzuschicken!")
        #st.markdown(
        #f'<button style="background-color: {button_styles_try["background-color"]}; '
        #f'color: {button_styles_try["color"]}; '
        #f'font-size: {button_styles_try["font-size"]}; '
        #f'padding: {button_styles_try["padding"]}; '
        #f'border-radius: {button_styles_try["border-radius"]};">Abschicken</button>',
        #unsafe_allow_html=True,
    #)
  
        
       
        
    

        empty_slot = st.markdown('')
    with col3:
        #st.image(logo1, width = 300)
        col4, col5 = st.columns([1,1])
        with col5:
            note = st.image(notice, width=300)

        
    st.markdown("</div>", unsafe_allow_html=True)



        
   
        


    # Create footer section
    footer_bg = f"background-color: #FFFFF; border-radius: 5px; padding: 10px; margin-top: 20px;"
    st.markdown(f"<div style='{footer_bg}'>", unsafe_allow_html=True)
    col0,col1, col2, col3, col4,colEnd = st.columns([2,1, 1,1,1,2])
    with col1:
        st.image(logoHSEL, width=160)
    with col2:
        st.image(logoEXIST, width=140)
    with col3:
        st.image(logoBMWK, width=160)
    with col4:
        st.image(logo2, width=160)
    st.markdown("</div>", unsafe_allow_html=True)
    
    output = ''
        # Check if the button is pressed
    if message:
       
        keyInt = keyInt + 1
        ai_question = message
        #Greife den Eintrag ab, der am meisten Änhlichkeit mit der Frage hat
        res = search_docs(df_try, ai_question, top_n=1)
        #Greife den Inhalt des Eintrages ab
        context= res.CONTENT.values
        
        #Kombiniere den Prompt mit Baisisprompt, dem Inhalt und der Frage
        combined_prompt = initial_prompt + str(context) + output + "Q: " + ai_question
            
        #API-Abfrage
        messages.append(
            {"role": "user", "content": combined_prompt},
        )        
        chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
        )
            
        #Greife Inhalt des Resultat der API Abfrage ab
        ai_response = chat.choices[0].message.content
        output = ai_response
        update_markdown(empty_slot,output)
     
        
        
    if button:
        keyInt = keyInt + 1
        ai_question = message
        #Greife den Eintrag ab, der am meisten Änhlichkeit mit der Frage hat
        res = search_docs(df_try, ai_question, top_n=1)
        #Greife den Inhalt des Eintrages ab
        context= res.CONTENT.values
        
        #Kombiniere den Prompt mit Baisisprompt, dem Inhalt und der Frage
        combined_prompt = initial_prompt + str(context) + output + "Q: " + ai_question
            
        #API-Abfrage
        messages.append(
            {"role": "user", "content": combined_prompt},
        )        
        chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
        )
            
        #Greife Inhalt des Resultat der API Abfrage ab
        ai_response = chat.choices[0].message.content
        output = ai_response
        update_markdown(empty_slot,output)
        message = ""
        
        
    if q1_btn:
        keyInt = keyInt + 1
        ai_question = Beispiel1
        #Greife den Eintrag ab, der am meisten Änhlichkeit mit der Frage hat
        res = search_docs(df_try, ai_question, top_n=1)
        #Greife den Inhalt des Eintrages ab
        context= res.CONTENT.values
        
        #Kombiniere den Prompt mit Baisisprompt, dem Inhalt und der Frage
        combined_prompt = initial_prompt + str(context) + output + "Q: " + ai_question
            
        #API-Abfrage
        messages.append(
            {"role": "user", "content": combined_prompt},
        )        
        chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
        )
            
        #Greife Inhalt des Resultat der API Abfrage ab
        ai_response = chat.choices[0].message.content
        output = ai_response
        update_markdown(empty_slot,output)
        
        
    if q2_btn:
        keyInt = keyInt + 1
        ai_question = Beispiel2
        #Greife den Eintrag ab, der am meisten Änhlichkeit mit der Frage hat
        res = search_docs(df_try, ai_question, top_n=1)
        #Greife den Inhalt des Eintrages ab
        context= res.CONTENT.values
        
        #Kombiniere den Prompt mit Baisisprompt, dem Inhalt und der Frage
        combined_prompt = initial_prompt + str(context) + output + "Q: " + ai_question
            
        #API-Abfrage
        messages.append(
            {"role": "user", "content": combined_prompt},
        )        
        chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
        )
            
        #Greife Inhalt des Resultat der API Abfrage ab
        ai_response = chat.choices[0].message.content
        output = ai_response
        update_markdown(empty_slot,output)
        
    if q3_btn:
        keyInt = keyInt + 1
        ai_question = Beispiel3
        #Greife den Eintrag ab, der am meisten Änhlichkeit mit der Frage hat
        res = search_docs(df_try, ai_question, top_n=1)
        #Greife den Inhalt des Eintrages ab
        context= res.CONTENT.values
        
        #Kombiniere den Prompt mit Baisisprompt, dem Inhalt und der Frage
        combined_prompt = initial_prompt + str(context) + output + "Q: " + ai_question
            
        #API-Abfrage
        messages.append(
            {"role": "user", "content": combined_prompt},
        )        
        chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
        )
            
        #Greife Inhalt des Resultat der API Abfrage ab
        ai_response = chat.choices[0].message.content
        output = ai_response
        update_markdown(empty_slot,output)




