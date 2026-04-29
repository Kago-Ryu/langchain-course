from dotenv import load_dotenv
from langchain.tools import tool_node
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode


from react import llm,tools

load_dotenv()

SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""


def run_agent_reasoning_engine(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    # 调用模型，传入系统消息和之前的对话消息，获取模型的回复，
    # 中state["messages"]是之前的对话消息列表，模型会根据这些消息来生成回复。
    # *表示将state["messages"]列表中的每个元素作为单独的参数传递给invoke方法。
    response = llm.invoke([{"role":"system","content": SYSTEM_MESSAGE},*state["messages"]])
    return {"messages":[response]}

tool_node = ToolNode(tools=tools)
