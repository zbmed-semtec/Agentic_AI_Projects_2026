"""
Dissify Exploration Agent: Agent Loop Implementation.
Adheres to official LangChain 0.3+ and LangGraph standards.
Uses LangGraph's prebuilt ReAct agent with fallback to native LangChain-Core tool binding.
"""

from collections import namedtuple
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    MAX_AGENT_STEPS,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
)
from tools import (
    resolve_historical_terms,
    search_dissertation_catalog,
    inspect_table_of_contents,
    read_page_ocr,
    inspect_author_timeline,
)

try:
    from langgraph.prebuilt import create_react_agent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

AgentAction = namedtuple("AgentAction", ["tool", "tool_input", "log"])

SYSTEM_PROMPT = """You are the Dissify Exploration Agent, an autonomous scientific archivist assisting researchers with historical European dissertations (1850–1950).

### Core Rules for Grounding & Integrity:
1. Never Guess or Hallucinate: You may ONLY state facts, numbers, dates, or findings that were directly returned by your tools.
2. Grounding & Provenance: Whenever citing findings, you MUST specify:
   - Document ID
   - Section Title
   - Page Number
   - Exact supporting quote or excerpt from the page text.
3. Refusal Protocol: If the Table of Contents does not contain a relevant section, or if the examined page text does not answer the question, explicitly state:
   "Refusal: Insufficient evidence found in the examined primary records." Do NOT invent an answer.
4. Hierarchical Strategy:
    - Step 1 (Terminology Resolution): If the question uses modern scientific terminology, first call `resolve_historical_terms` to identify the corresponding historical terminology or concepts used in European dissertations of the relevant period.
        - If `resolve_historical_terms` returns a match, use the returned historical terminology to guide the subsequent catalog search and document investigation.
        - If `resolve_historical_terms` returns no match, you MAY use your own general historical, linguistic, and scientific knowledge to infer plausible historical German terminology equivalents that could correspond to the modern term.
        - Treat such knowledge-based terminology mappings as search hypotheses only, not as evidence about what a particular dissertation contains or means.
        - Clearly distinguish between:
            - Terminology hypotheses generated from your own knowledge, and
            - Verified historical evidence returned by the primary-record tools.
        - Do NOT cite or present a knowledge-based terminology mapping as a historical finding unless it is subsequently supported by the examined primary records.
        - If neither the terminology tool nor your own knowledge provides a plausible mapping, continue using the original terminology and available search strategies rather than inventing a mapping.
   - Step 2 (Catalog Filter): Call `search_dissertation_catalog` to identify the most relevant document ID.
   - Step 3 (Structural TOC Navigation): Call `inspect_table_of_contents` for that document to find the exact chapter and page range.
   - Step 4 (Deep Read): Call `read_page_ocr` to read only the specific target page(s). If you think you need to read more pages You could try as well of those specific document.
            - If you encounter only with "Textanfang" section in table of content, then read first and last 3 pages to retrieve information for analysis.
   - Step 5 (Author Details): If the question asks about author background, biographical details, author timeline or vital dates, call `inspect_author_timeline`.

5. Bounded Autonomy: Be efficient and targeted. Do not read excessive pages. Stop and answer as soon as you have verified evidence.
"""


class DissifyAgentExecutor:
    def __init__(
        self,
        model_name: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        max_steps: int = MAX_AGENT_STEPS,
        recursion_limit: int = None,
        num_ctx: int = OLLAMA_NUM_CTX,
        num_predict: int = OLLAMA_NUM_PREDICT,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.max_steps = max_steps
        self.recursion_limit = recursion_limit or max(50, max_steps * 4)
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.tools = [
            resolve_historical_terms,
            search_dissertation_catalog,
            inspect_table_of_contents,
            read_page_ocr,
            inspect_author_timeline,
        ]
        self.tools_map = {t.name: t for t in self.tools}

        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.0,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
        )

        self.graph = None
        if LANGGRAPH_AVAILABLE:
            try:
                self.graph = create_react_agent(
                    model=self.llm,
                    tools=self.tools,
                    prompt=SYSTEM_PROMPT
                )
            except (TypeError, Exception):
                # try:
                #     # Fallback to LangGraph 'state_modifier' parameter
                #     self.graph = create_react_agent(
                #         model=self.llm,
                #         tools=self.tools,
                #         state_modifier=SYSTEM_PROMPT
                #     )
                # except Exception:
                #     self.graph = None
                pass
                

        # Fallback engine: Native LangChain-Core Tool Binding
        if self.graph is None:
            self.llm_with_tools = self.llm.bind_tools(self.tools)

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_query = inputs.get("input", "")

        # Preferred LangGraph Path
        if self.graph is not None:
            try:
                result = self.graph.invoke(
                    {"messages": [HumanMessage(content=user_query)]},
                    config={"recursion_limit": self.recursion_limit}
                )
                messages = result.get("messages", [])
                intermediate_steps = []

                # Extract intermediate actions and observations from graph message history
                for i, msg in enumerate(messages):
                    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            # Look for matching ToolMessage in subsequent messages
                            obs = ""
                            for sub_msg in messages[i + 1:]:
                                if isinstance(sub_msg, ToolMessage) and getattr(sub_msg, "tool_call_id", "") == tc.get("id"):
                                    obs = sub_msg.content
                                    break
                            action = AgentAction(tool=tc["name"], tool_input=tc.get("args", {}), log=str(tc))
                            intermediate_steps.append((action, obs))

                final_output = messages[-1].content if messages else "No response generated."
                return {
                    "output": final_output,
                    "intermediate_steps": intermediate_steps
                }
            except Exception as e:
                pass

        # Fallback Native LangChain loop
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_query)
        ]
        intermediate_steps = []

        for _ in range(self.max_steps):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            if not getattr(response, "tool_calls", None):
                return {
                    "output": response.content,
                    "intermediate_steps": intermediate_steps
                }

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args", {})
                tool_id = tc.get("id", tool_name)

                tool_obj = self.tools_map.get(tool_name)
                if tool_obj:
                    try:
                        observation = tool_obj.invoke(tool_args)
                    except Exception as err:
                        observation = f"Error executing tool '{tool_name}': {err}"
                else:
                    observation = f"Error: Tool '{tool_name}' is not recognized."

                action = AgentAction(tool=tool_name, tool_input=tool_args, log=str(tc))
                intermediate_steps.append((action, observation))

                messages.append(
                    ToolMessage(
                        content=str(observation),
                        tool_call_id=tool_id
                    )
                )

        return {
            "output": "Refusal: Maximum reasoning steps reached without concluding evidence.",
            "intermediate_steps": intermediate_steps
        }


def create_dissify_agent(
    model_name: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    max_steps: int = MAX_AGENT_STEPS,
    recursion_limit: int = None,
    num_ctx: int = OLLAMA_NUM_CTX,
    num_predict: int = OLLAMA_NUM_PREDICT,
):
    """Initializes the Dissify Exploration Agent connected to local Ollama."""
    return DissifyAgentExecutor(
        model_name=model_name,
        base_url=base_url,
        max_steps=max_steps,
        recursion_limit=recursion_limit,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )
