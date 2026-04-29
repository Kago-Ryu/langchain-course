from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def tripe(num:float) -> float:
    """
    param num:  a number to tripe
    returns : the tripe of the number
    """
    return float(num) * 3

# 设定工具列表的时候，工具的顺序会影响模型的调用，模型会优先调用列表前面的工具，
# 所以如果你想让模型优先使用TavilySearch工具，就把它放在tripe前面。
tools = [TavilySearch(max_results=1),tripe]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


