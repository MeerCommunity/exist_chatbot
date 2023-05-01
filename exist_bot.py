#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import time
import openai

def generate_response(user_input):
    # 用您的OpenAI API密钥进行身份验证
    openai.api_key = "your_openai_api_key_here"

    # 设置GPT-3模型的相关参数
    model_engine = "text-davinci-002"  # 或其他GPT-3模型，如"gpt-3.5-turbo"
    prompt = f"用户：{user_input}\nChatbot："
    max_tokens = 100

    # 调用OpenAI API生成回答
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # 提取并返回生成的文本
    return response.choices[0].text.strip()



st.title("IPRO-Demo")

# 创建一个输入框，用于接收用户输入
user_input = st.text_input("Frage Hier：")

if st.button("Send"):
    # 模拟ChatBot正在生成回答
    with st.spinner("Waiting..."):
        time.sleep(2)

    # 调用generate_response函数生成回答
    response = generate_response(user_input)

    # 显示回答
    st.write(response)
