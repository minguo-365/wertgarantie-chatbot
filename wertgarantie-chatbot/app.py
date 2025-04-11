
import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from llama_index import SimpleDirectoryReader, ServiceContext, GPTVectorStoreIndex
from llama_index.llms import OpenAI

# 在代码中自动安装必要的包
def install_packages():
    required_packages = [
        'llama_index', 
        'openai', 
        'beautifulsoup4', 
        'requests', 
        'streamlit'
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + required_packages)

install_packages()

# 配置 OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# 设置爬取数据的文件路径
file_path = "/Users/YourUsername/Documents/wertgarantie-chatbot/data/wertgarantie.txt"  # 请替换为你的用户目录

# 创建文件夹（如果不存在）
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# 网页爬取
url = "https://www.wertgarantie.de"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# 获取网页文本内容
text = soup.get_text()

# 保存为文本文件
with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

# 输出文件保存路径
st.write(f"文件已保存到： {file_path}")

# 加载爬取的数据
docs = SimpleDirectoryReader("data").load_data()

# 设置 OpenAI LLM 模型
llm = OpenAI(model="gpt-3.5-turbo")

# 创建服务上下文
service_context = ServiceContext.from_defaults(llm=llm)

# 创建索引
index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)

# 创建流式界面
st.set_page_config(page_title="Wertgarantie Chatbot", layout="wide")

st.markdown("""
    <h1 style='text-align: center;'>🤖 Wertgarantie 智能客服助手</h1>
    <p style='text-align: center;'>
    培训自 Wertgarantie 网站的内容，支持中文和德语对话 🇩🇪 🇨🇳
    </p>
    """, unsafe_allow_html=True)

# 用户输入框
user_input = st.text_input("请输入您的问题:")

if user_input:
    response = index.query(user_input)
    st.write(f"答复: {response}")

