# agent.py
import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from wine_tool import predict_wine_quality

# 加载 .env 文件中的环境变量
load_dotenv()

# 设置通义千问密钥（如果你已经在 .env 中设置了，这行可以注释掉）
# 但为了保险，我们直接在代码里设置
#os.environ["DASHSCOPE_API_KEY"] = ""
# 初始化大模型（通义千问）
llm = ChatTongyi(model="qwen-plus", temperature=0)

# 定义工具列表
tools = [predict_wine_quality]

# 定义 ReAct 提示模板（这是告诉智能体如何思考的格式）
template = """你是一个助手，可以调用工具来回答问题。你有以下工具可用：

{tools}

使用格式：
Question: 用户的问题
Thought: 思考需要做什么
Action: 工具名称，必须是 [{tool_names}] 中的一个
Action Input: 工具的输入，应为一个JSON对象（包含所有必需的参数）
Observation: 工具返回的结果
...（重复 Thought/Action/Action Input/Observation 直到得到答案）
Thought: 我现在知道答案了
Final Answer: 对用户的回答

开始！

Question: {input}
Thought: {agent_scratchpad}"""

# 创建提示模板对象
prompt = PromptTemplate.from_template(template)

# 创建智能体
agent = create_react_agent(llm, tools, prompt)

# 创建执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

if __name__ == "__main__":
    # 测试输入数据（一组真实红酒指标）
    test_input = """固定酸度7.4，挥发性酸度0.7，柠檬酸0.0，残糖1.9，
    氯化物0.076，游离二氧化硫11，总二氧化硫34，密度0.9978，pH3.51，
    硫酸盐0.56，酒精9.4，帮我评估一下这款红酒的品质。"""
    
    print("🤖 正在呼叫 AI 智能体...\n")
    result = agent_executor.invoke({"input": test_input})
    print("\n" + "="*50)
    print("📝 最终回答：")
    print(result["output"])
    print("="*50)