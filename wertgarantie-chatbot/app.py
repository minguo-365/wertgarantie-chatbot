import streamlit as st
from llama_index import StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorIndexRetriever

INDEX_DIR = "index"

st.set_page_config(
    page_title="Wertgarantie Chatbot", layout="wide"
)

st.markdown("""
    <h1 style='text-align: center;'>🤖 Wertgarantie 智能助手</h1>
    <p style='text-align: center;'>培训于 Wertgarantie 官方 FAQ，支持用户问题答复</p>
""", unsafe_allow_html=True)

# 加载向量索引
@st.cache_resource(show_spinner=True)
def load_index():
    llm = OpenAI(model="gpt-4o")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)
    return index

index = load_index()
retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
query_engine = RetrieverQueryEngine(retriever=retriever)

# 用户输入
user_input = st.chat_input("请输入你的问题...")

if user_input:
    st.chat_message("user").markdown(user_input)
    with st.spinner("AI 回复中..."):
        response = query_engine.query(user_input)
        st.chat_message("assistant").markdown(response.response)
