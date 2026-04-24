# 3 号文件：最原始的 ReAct 代理写法（无 LangChain、无 function calling）。
# 对照 README：这一层把“工具调用协议”从 API 层，退回到“纯提示词 + 文本解析”。

# re: 用正则从模型输出文本中提取 Action / Action Input / Final Answer。
import re
# inspect: 读取函数签名和文档，把工具说明自动拼进提示词。
import inspect
from dotenv import load_dotenv

# 读取 .env（例如 LangSmith key）。
load_dotenv()

import ollama
from langsmith import traceable

# 防止循环失控。
MAX_ITERATIONS = 10
MODEL = "qwen3.5:4b"


# --- Tools ---
# 这里不使用 @tool（LangChain）和 tools=JSON schema（function calling），
# 只是普通 Python 函数 + 手工调度。


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """根据商品名查询目录价格。"""
    print(f"    >> Executing get_product_price(product='{product}')")
    # 真实项目通常来自数据库或外部 API，这里用字典模拟。
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """按折扣等级计算最终价格，可用等级：bronze/silver/gold。"""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    # ReAct 场景里参数来自文本解析，先转成 float 提升容错性。
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# 工具注册表：模型说出工具名后，我们据此找到对应函数并执行。
tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

# 对照 2 号文件：2 里工具描述在 tools_for_llm JSON schema；
# 3 里工具描述直接写进 prompt 文本（模型只看到文字，不知道有“结构化工具协议”）。

def get_tool_descriptions(tools_dict):
    # 把工具函数动态转成可读文本：
    # get_product_price(product: str) -> float - 根据商品名查询目录价格。
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ 可绕过装饰器包装层，拿到原始函数签名。
        # 否则 @traceable 可能让签名出现额外参数，影响提示词清晰度。
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        # 没有 docstring 时回退为空字符串，避免拼接时报错。
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)


# 这两行会被插入 ReAct 提示词里，告诉模型“可用工具”和“工具名字列表”。
tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

# ReAct 核心提示词：
# 1) 先定义规则（必须先查价、再打折、不能心算）
# 2) 再定义输出格式（Thought/Action/Action Input/Observation/Final Answer）
# 3) 最后让模型从 Thought 开始接着写
#
# 对照 1/2 号文件：
# - 1/2 把“工具协议”交给框架/API（结构化 tool_calls）
# - 3 把“工具协议”写成自然语言约定（更原始，也更脆弱）
react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


# 对照 2 号文件：这里没有 tools= 参数。
# 也就是说，模型 API 不会返回结构化 tool_calls；
# “代理能力”全部来自提示词约束 + 我们自己的正则解析器。

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    # 仍然是普通 chat，只是消息内容是“完整 ReAct 提示词 + 历史 scratchpad”。
    return ollama.chat(model=model, messages=messages, options=options)





# --- Agent Loop ---


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)


    # 对照 1/2：1/2 维护 message 列表（system/user/assistant/tool）；
    # 3 使用单一 prompt 模板 + scratchpad 字符串。
    prompt = react_prompt.format(question=question)
    # scratchpad 会累积每一轮的 Thought/Action/Observation，
    # 下一轮把整段重新发给模型，模拟“记忆”。
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        # 每轮把“初始模板 + 历史轨迹”拼成完整输入。
        full_prompt = prompt + scratchpad

        # stop 的作用：在模型准备输出 Observation 前截断。
        # 这样 Observation 由真实工具执行结果填充，而不是模型瞎编。
        # 这是 ReAct 纯文本代理中非常关键的一步。
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.message.content
        print(f"LLM Output:\n{output}")

        # Finish 阶段：若模型已经输出 Final Answer，流程结束。
        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        # Parse 阶段：从“纯文本”提取 Action / Action Input。
        # 对照 1/2：1/2 是结构化字段读取；3 是 regex，格式稍偏就会解析失败。
        print(f"  [Parsing] Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

        # Execute 前做参数预处理：
        # - 按逗号切分参数
        # - 支持 key=value 或纯值两种格式
        # - 去掉引号，便于直接传入函数
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))


        print(f"  [Tool Result] {observation}")

        # Observe 阶段：把本轮输出与真实 Observation 拼回 scratchpad。
        # 下一轮模型会看到完整轨迹并继续写 Thought。
        # 对照 1/2：1/2 是 append 消息对象/字典；3 是 append 字符串。
        scratchpad += f"{output}\nObservation: {observation}\nThought:"


    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    # 运行演示：固定问题，便于对照 1/2 结果。
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")