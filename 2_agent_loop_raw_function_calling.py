from dotenv import load_dotenv

# 读取 .env 配置，确保 LangSmith / 其他环境变量可用。
load_dotenv()

# 纯 Ollama 版本：直接调用 ollama.chat，不走 LangChain 封装。
import ollama
# 这里保留 LangSmith 追踪能力，用来观察每轮模型与工具调用。
from langsmith import traceable

# 最大循环次数，防止异常情况下无限循环。
MAX_ITERATIONS = 10
# 使用的 Ollama 模型。
MODEL = "qwen3.5:4b"


# ------ Tools(LangChain @tool decorator)-------
@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """根据商品名查询目录价格。"""
    print(f"   >> Executing get_product_price(product='{product}')")
    # 模拟商品数据源。
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """按折扣等级计算最终价格。可用等级：bronze/silver/gold。"""
    print(
        f"   >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# =============================
# 与 1 号文件（LangChain 版）的核心差异 A：工具定义方式
# -----------------------------
# 1 号文件中：
#   - 用 @tool 装饰器后，LangChain 会自动把函数转成可被模型调用的 schema。
# 2 号文件中（当前文件）：
#   - 我们不依赖 LangChain 的 @tool 自动转换。
#   - 必须手动写出每个函数的 JSON schema（名字、描述、参数类型、必填项）。
# =============================
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# 说明：Ollama 也支持把函数直接传进去自动推断 schema，示意如下：
#   tools_for_llm = [get_product_price, apply_discount]
# 但 docstring 需要按 Google 风格写 Args/Returns，解析才更稳定。例如：
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# 这里保留“手写 JSON”是为了教学：能看清 @tool 背后到底做了什么。

# --- Helper: traced Ollama call ---
# 与 1 号文件（LangChain 版）的核心差异 B：模型调用方式
# -----------------------------
# 1 号文件中：llm_with_tools.invoke(messages)
# 2 号文件中：ollama.chat(model=..., tools=..., messages=...)
# 也就是我们直接面对 Ollama 的原生接口和返回结构。


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    # 纯 Ollama：手动传入 model / tools / messages。
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# ----- Agent Loop -----

# 对照 README：这是“去掉 LangChain，但仍保留模型原生 function calling”的中间层。
# 也就是你要手动处理 schema、消息格式和工具分发，但还不用正则解析文本。


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    # 函数名 -> Python 函数对象，用于收到 tool call 后执行本地函数。
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 20)

    # 与 1 号文件差异 C：消息格式
    # -----------------------------
    # 1 号文件用的是 LangChain 的消息对象：SystemMessage/HumanMessage/ToolMessage
    # 2 号文件用的是 Ollama 原生字典：{"role": "...", "content": "..."}
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant."
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
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    # Agent 循环（对应 README 五阶段）:
    # - Reason: ollama.chat(...) 产出结构化 tool_calls
    # - Parse:  读取 tool_call.function.name / arguments
    # - Execute: Python 函数调用 tool_to_use(**tool_args)
    # - Observe: 追加 role=tool 的消息字典
    # - Finish: 无 tool_calls 时返回最终答案
    # 对比 3 号文件：3 没有结构化 tool_calls，Parse 要靠 regex。
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n ---- Iteration {iteration} ----   ")

        # 直接调用 Ollama（非 LangChain 封装）。
        response = ollama_chat_traced(messages=messages)

        # 注意：Ollama 返回结构和 LangChain 不同，这里拿的是 response.message。
        ai_message = response.message

        # 若模型想调用工具，tool_calls 会包含函数名与参数。
        tool_calls = ai_message.tool_calls

        # 没有 tool_calls 说明模型已给出最终自然语言答案。
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        # 教学简化：每轮只处理第一个工具调用，流程更容易跟踪。
        tool_call = tool_calls[0]

        # 与 1 号文件差异 D：tool_call 取值方式
        # -----------------------------
        # 1 号文件（LangChain）通常是 dict 风格：tool_call.get("name")
        # 2 号文件（Ollama）是对象字段：tool_call.function.name
        # 3 号文件（ReAct）则没有这个对象结构，只能从 "Action:" 文本行提取。
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f" [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        # 与 1 号文件差异 E：工具执行方式
        # -----------------------------
        # 1 号文件：tool_to_use.invoke(tool_args)
        # 2 号文件：普通 Python 函数调用 tool_to_use(**tool_args)
        observation = tool_to_use(**tool_args)

        print(f" [Tool Observation] {observation}")

        # 与 1 号文件差异 F：回填工具结果消息格式
        # -----------------------------
        # 1 号文件追加 ToolMessage(content=..., tool_call_id=...)
        # 2 号文件追加原生 role=tool 的字典。
        # 这里没有 tool_call_id，是因为我们直接遵循 Ollama 消息格式。
        # 对比 3 号文件：3 完全不维护消息列表，而是把 Observation 追加到 scratchpad 字符串。
        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    print("Error: Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    # 直接运行脚本时，执行一个固定问题示例。
    print("Hello LangChain Agent!")
    print()
    result = run_agent("What is the price of a laptop with a gold discount?")
