# chat_app.py
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from wine_tool import predict_wine_quality

# ============================================================
# 1. 加载 .env 文件中的环境变量（DASHSCOPE_API_KEY）
# ============================================================
load_dotenv()

# 检查密钥是否加载成功（如果没加载成功，程序会报错）
if not os.getenv("DASHSCOPE_API_KEY"):
    st.error("❌ 未找到 DASHSCOPE_API_KEY，请在 .env 文件中设置你的通义千问密钥！")
    st.stop()

# ============================================================
# 2. 页面配置
# ============================================================
st.set_page_config(page_title="🍷 AI 品酒助手", page_icon="🍷")
st.title("🍷 AI 品酒助手")
st.markdown("在下方输入你的问题，AI 会**自动判断**要不要调用工具来回答你")

# ============================================================
# 3. 初始化 Agent（用缓存，避免每次对话都重新加载）
# ============================================================
@st.cache_resource
def get_agent():
    llm = ChatTongyi(
        model="qwen-plus",
        temperature=0,
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")  # 显式传入密钥
    )
    
    tools = [predict_wine_quality]
    
    template = """你是一个葡萄酒品鉴专家助手。你有以下工具可用：

{tools}

使用格式：
Question: 用户的问题
Thought: 思考需要做什么
Action: 工具名称，必须是 [{tool_names}] 中的一个
Action Input: 工具的输入，必须是包含所有11个参数的JSON对象
Observation: 工具返回的结果
...（重复 Thought/Action/Action Input/Observation 直到得到答案）
Thought: 我现在知道答案了
Final Answer: 对用户的回答

开始！

Question: {input}
Thought: {agent_scratchpad}"""
    
    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

agent = get_agent()

# ============================================================
# 4. 聊天界面
# ============================================================

# 保存聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 输入框（固定在页面底部）
user_input = st.chat_input("输入你的问题...")

if user_input:
    # 显示用户消息
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 调用 Agent
    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            try:
                response = agent.invoke({"input": user_input})
                st.write(response["output"])
                st.session_state.messages.append({"role": "assistant", "content": response["output"]})
            except Exception as e:
                st.error(f"出错了：{str(e)}")