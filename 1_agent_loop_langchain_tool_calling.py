from dotenv import load_dotenv

# 读取 .env 文件中的环境变量（例如 LangSmith 追踪相关配置）。
# 放在其他依赖初始化前执行，确保后续库能拿到配置。
load_dotenv()

# init_chat_model: 创建一个聊天模型实例（这里用的是 Ollama 后端）。
from langchain.chat_models import init_chat_model
# @tool: 把普通 Python 函数包装成 LangChain Tool，
# LangChain 会基于类型标注和 docstring 自动生成工具 schema。
from langchain.tools import tool
# 三种消息类型：用户消息、系统消息、工具结果消息。
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
# @traceable: 把函数执行过程发送到 LangSmith，便于调试和可视化。
from langsmith import traceable

# 为了避免模型一直循环调用工具，设置最大迭代次数。
MAX_ITERATIONS = 10
# 指定本地 Ollama 模型名。
MODEL = "qwen3.5:4b"


# ------ Tools(LangChain @tool decorator)-------
@tool
def get_product_price(product: str) -> float:
    """根据商品名查询目录价格。"""
    # 这里用 print 只是为了教学：你能在终端看到工具是否真的被调用。
    print(f"   >> Executing get_product_price(product='{product}')")
    # 模拟一个商品数据库（真实项目通常会查数据库/接口）。
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    # 未命中时返回 0，避免抛异常中断流程。
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """按折扣等级计算最终价格。可用等级：bronze/silver/gold。"""
    print(f"   >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    # 不同会员等级对应不同折扣百分比。
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    # 若等级不合法，默认 0% 折扣。
    discount = discount_percentages.get(discount_tier, 0)
    # round(..., 2) 保留两位小数，符合金额展示习惯。
    return round(price * (1 - discount / 100), 2)


# ----- Agent Loop -----

# 对照 README：这是三种实现里“抽象层最高”的版本。
# 你主要写业务函数，LangChain 帮你处理大部分工具调用细节。

# 整个 Agent 主流程也做追踪，方便在 LangSmith 中查看每次循环。
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    # LangChain Tool 对象列表。
    tools = [get_product_price, apply_discount]
    # 通过工具名快速查找具体函数，后面执行工具时会用到。
    tools_dict = {t.name: t for t in tools}

    # 创建聊天模型（temperature=0 表示更稳定、可复现，适合教学）。
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    # 关键点：把工具“绑定”到模型。
    # 之后模型输出里可以带 tool_calls，告诉我们它想调用哪个工具。
    # README 对照：这一步对应 File 2 里手写 tools_for_llm JSON schema，
    # 也对应 File 3 里把工具描述塞进 prompt 文本。
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 20)

    # 对话上下文（messages）会在循环中不断追加：
    # 系统提示 -> 用户问题 -> 模型消息 -> 工具消息 -> 模型消息 ...
    messages = [
        SystemMessage(
            content="You are a helpful shopping assistant."
            "You have access to a product catalog tool "
            "and a discount tool.\n\n"
            "STRICT RULES - you must follow these exactly:\n"
            "1. NEVER guess or assume any product price."
            "You MUST call get_product_price first to get the real price.\n"
            "2. Only call apply_discount AFTER you have received "
            "a price from get_product_price. Pass the exact price "
            "returned by get_product_price - do NoT pass a made-up number. \n"
            "3. Never calculate discount yourself using math"
            "Always use the apply_discount tool."
            "4. If the user does not specify a discounttier,"
            "ask them which tier to use - do NOT assume one."
        ),
        HumanMessage(content=question),
    ]

    # Agent Loop: 每轮做一次“模型思考 -> (可选)工具调用 -> 观察结果 -> 回填上下文”。
    # README 对照到 5 个阶段：
    # - Reason: llm_with_tools.invoke(messages)
    # - Parse:  ai_message.tool_calls
    # - Execute: tool.invoke(args)
    # - Observe: ToolMessage(...)
    # - Finish: 无 tool_calls 时直接返回
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n ---- Iteration {iteration} ----   ")

        # 让模型基于当前上下文决定：是直接回答，还是继续调工具。
        ai_message = llm_with_tools.invoke(messages)

        # tool_calls 是 LangChain 已统一好的结构（list[dict]）。
        # 对比 2 号文件：2 里是 Ollama 原生对象字段（tool_call.function.name）。
        # 对比 3 号文件：3 里完全没有结构化 tool_calls，要自己用正则从文本里抠 Action。
        tool_calls = ai_message.tool_calls

        # 若没有工具调用，说明模型认为可以直接给最终答案。
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        # 教学简化：每轮只执行第一个工具调用。
        # 这样流程最清晰：一次只做一步，便于观察。
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        # tool_call_id 要回填到 ToolMessage 里，模型才能对上“哪次调用的返回值”。
        tool_call_id = tool_call.get("id")

        print(f" [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        # LangChain Tool 执行方式：tool.invoke(dict_args)
        observation = tool_to_use.invoke(tool_args)

        print(f" [Tool Observation] {observation}")

        # 把“模型刚才的工具调用请求”与“工具执行结果”都追加到上下文。
        # 下一轮模型就能基于这个 observation 继续推理。
        # 对比 2 号文件：2 使用原生 {"role": "tool"} 字典。
        # 对比 3 号文件：3 使用 scratchpad 字符串拼接 Observation。
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("Error: Max iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    # 直接运行本文件时，做一个固定问题的演示。
    print("Hello LangChain Agent!")
    print()
    result = run_agent("What is the price of a laptop with a gold discount?")
