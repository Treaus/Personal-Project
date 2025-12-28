# 导入必要的库
from langchain.chat_models import ChatOpenAI  # LLM 接口（这里用 OpenAI，可替换）
from langchain.prompts import ChatPromptTemplate  # Prompt 模板
from langchain.chains import LLMChain  # 基础 Chain
from langchain.memory import ConversationBufferMemory  # 会话记忆模块
from langchain.tools import Tool  # 自定义工具接口
from langchain.agents import initialize_agent, AgentType  # Agent 初始化
from langchain.vectorstores import FAISS  # 向量数据库
from langchain.embeddings import OpenAIEmbeddings  # 向量化模型
from langchain.text_splitter import CharacterTextSplitter  # 文本切分
from langchain.docstore.document import Document  # 文档对象
from langchain.chains import RetrievalQA  # 检索问答链

# 1. 准备基础 LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # 模型
    temperature=0,               # 生成稳定性
    openai_api_key="你的OpenAI API Key"
)

# 2. 准备 Prompt 模板
prompt_template = ChatPromptTemplate.from_template(
    "你是一个专业知识助手，请用简洁的语言回答问题：{question}"
)

# 3. 创建一个最简单的 LLMChain
simple_chain = LLMChain(
    llm=llm,                      # 使用的模型
    prompt=prompt_template,       # 提示词模板
    verbose=True                  # 显示执行日志
)

# 4. 添加记忆功能
memory = ConversationBufferMemory(
    memory_key="chat_history",    # 存储历史记录的键名
    return_messages=True          # 返回消息格式，便于 Agent 处理
)

# 5. 准备一个向量数据库（这里用 FAISS）
# 假设我们有一段知识库文本
knowledge_text = """
LangChain 是一个用于构建基于大语言模型的应用的开发框架。
它提供了 Prompt 管理、Chain 组合、Agent 工具调用、Memory 对话记忆、
以及与外部知识库集成等功能。
"""

# 切分文本
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
docs = text_splitter.split_text(knowledge_text)

# 转成 Document 对象
documents = [Document(page_content=t) for t in docs]

# 创建 Embeddings
embeddings = OpenAIEmbeddings(openai_api_key="你的OpenAI API Key")

# 存入 FAISS
vectorstore = FAISS.from_documents(documents, embeddings)

# 6. 创建一个 RetrievalQA（从向量库检索回答）
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"  # 简单拼接型检索
)

# 7. 定义一个外部工具（例如计算器）
def calculator_tool(input_str: str) -> str:
    """一个简单的加法计算器，输入格式：数字1+数字2"""
    try:
        nums = input_str.split("+")
        result = float(nums[0]) + float(nums[1])
        return str(result)
    except:
        return "输入格式错误，示例：2+3"

calc_tool = Tool(
    name="Calculator",
    func=calculator_tool,
    description="输入形如 '数字1+数字2'，返回两数之和"
)

# 8. 初始化 Agent
agent = initialize_agent(
    tools=[calc_tool, retrieval_chain],  # 工具（可以是 QA 检索、计算器等）
    llm=llm,                              # 核心 LLM
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  # 对话型 Agent
    memory=memory,                        # 上下文记忆
    verbose=True                          # 显示过程
)

# 9. 运行测试
print("=== 测试 1：普通问答 ===")
response1 = agent.run("LangChain 是什么？")
print("Agent 回答：", response1)

print("\n=== 测试 2：调用计算器 ===")
response2 = agent.run("帮我算一下 3+5")
print("Agent 回答：", response2)

print("\n=== 测试 3：记忆上下文 ===")
response3 = agent.run("它能做什么？")  # 上下文的“它”指 LangChain
print("Agent 回答：", response3)