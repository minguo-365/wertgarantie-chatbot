import streamlit as st
import os
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.service_context import ServiceContext

# 【配置 API Key 】
openai.api_key = os.getenv("OPENAI_API_KEY")

# 【数据读取和索引】
docs = SimpleDirectoryReader("data").load_data()
llm = OpenAI(model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)
index = VectorStoreIndex.from_documents(docs, service_context=service_context)

# 【设置 Streamlit UI 】
st.set_page_config(page_title="Wertgarantie Chatbot", layout="wide")

st.markdown("""
    <h1 style='text-align: center;'>🧑‍💻 Wertgarantie 智能客服助手</h1>
    <p style='text-align: center;'>
        培训自 Wertgarantie 网站的内容，支持中文和德语对话 🇩🇪 🇹🇼
    </p>
""", unsafe_allow_html=True)

# 【输入窗口】
user_input = st.text_input("请输入您的问题：")

# 【处理回复】
if user_input:
    query_engine = index.as_query_engine()
    response = query_engine.query(user_input)
    st.markdown(f"**答复：** {response.response}")
