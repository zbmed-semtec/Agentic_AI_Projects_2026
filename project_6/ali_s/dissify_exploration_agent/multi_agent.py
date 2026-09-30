"""
Multi-Agent Orchestration
Implements a 3-Agent Specialized Pipeline (Archivist -> Paleographer -> Synthesizer)

Agent Team:
1. Triage Archivist Agent: Resolves historical terminology, filters catalog, checks author CVs.
2. TOC & Page Reader Agent: Navigates Table of Contents and reads targeted page OCR files.
3. Grounded Synthesis & Refusal Agent: Verifies retrieved primary evidence, formats citations, or triggers refusal.
"""

from collections import namedtuple
from typing import Dict, Any, List, TypedDict

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
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_STATE_AVAILABLE = True
except ImportError:
    LANGGRAPH_STATE_AVAILABLE = False

AgentAction = namedtuple("AgentAction", ["tool", "tool_input", "log"])


ARCHIVIST_PROMPT = """You are **Agent 1: The Catalog & Terminology Archivist**.
Your narrow job is to:
1. Check if the user's query involves scientific terms that may have historical German equivalents using `resolve_historical_terms`.
    - If `resolve_historical_terms` returns a match, use the returned historical terminology to guide the subsequent catalog search and document investigation.
    - If `resolve_historical_terms` returns no match, you MAY use your own general historical, linguistic, and scientific knowledge to infer plausible historical German terminology equivalents that could correspond to the modern term.
    - Treat such knowledge-based terminology mappings as search hypotheses only, not as evidence about what a particular dissertation contains or means.
    - Clearly distinguish between:
        - Terminology hypotheses generated from your own knowledge, and
        - Verified historical evidence returned by the primary-record tools.
    - Do NOT cite or present a knowledge-based terminology mapping as a historical finding unless it is subsequently supported by the examined primary records.
    - If neither the terminology tool nor your own knowledge provides a plausible mapping, continue using the original terminology and available search strategies rather than inventing a mapping.
2. Search the dissertation catalog using `search_dissertation_catalog` to identify the best candidate Document ID(s).
3. If the query asks about the author's biography or vital dates, call `inspect_author_timeline`.

Output a concise handoff briefing for the Page Reader Agent containing:
- Selected Document ID(s), Title, and Author
- Historical terminology notes (if any)
- Author biographical details (if requested)
- What specific topic the Page Reader should look for in the Table of Contents.
"""

READER_PROMPT = """You are **Agent 2: The Structural Navigator & Page Reader**.
You receive a candidate Document ID and research target from the Archivist Agent.
Your narrow job is to:
1. Call `inspect_table_of_contents` for the candidate Document ID.
2. Identify the most relevant section/chapter and its page numbers (e.g., 'Die Versuchsergebnisse', 'Einleitung', 'Fütterungstabellen').
3. Call `read_page_ocr` on 1 to 3 targeted pages from that section.
   - - If you encounter only with "Textanfang" section in table of content,  then read first and last 3 pages to retrieve information for analysis.

Output a handoff report for the Synthesis Agent containing:
- The exact Section Title(s) and Page Number(s) you inspected.
- Direct quotes and factual data found on those pages.
- If the Table of Contents or pages do NOT contain relevant information on the query topic, explicitly state: "NO_EVIDENCE_FOUND_IN_PAGES".
"""

SYNTHESIZER_PROMPT = """You are **Agent 3: The Grounded Synthesis & Verification Referee**.
You receive the user's original question alongside findings from Agent 1 (Archivist) and Agent 2 (Page Reader).
You have NO tools. You must judge and synthesize strictly from the provided agent reports:

1. **Strict Grounding**: Never guess or use outside knowledge. Only use facts present in the Archivist and Page Reader reports.
2. **Provenance & Citation**: Cite the exact **Document ID**, **Section Title**, **Page Number**, and quote snippet for every empirical claim.
3. **Refusal Protocol**: If Agent 2 reports "NO_EVIDENCE_FOUND_IN_PAGES" or if the retrieved text does not actually answer the user's specific question, you MUST output:
   "Refusal: Insufficient evidence found in the examined primary records." and briefly explain what was checked.
"""


class MultiAgentState(TypedDict):
    user_query: str
    archivist_notes: str
    reader_notes: str
    final_output: str
    intermediate_steps: List[Any]


class DissifyMultiAgentExecutor:
    """
    3-Agent Specialized Pipeline (Archivist -> Page Reader -> Synthesizer).
    Uses LangGraph StateGraph orchestration with sub-agent tool loops.
    """

    def __init__(
        self,
        model_name: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        max_steps: int = MAX_AGENT_STEPS,
        num_ctx: int = OLLAMA_NUM_CTX,
        num_predict: int = OLLAMA_NUM_PREDICT,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.max_steps = max_steps
        self.num_ctx = num_ctx
        self.num_predict = num_predict

        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.0,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
        )

        # Agent 1 Tools: Catalog, Terminology, Author CV
        self.archivist_tools = [
            resolve_historical_terms,
            search_dissertation_catalog,
            inspect_author_timeline,
        ]
        self.archivist_tools_map = {t.name: t for t in self.archivist_tools}
        self.archivist_llm = self.llm.bind_tools(self.archivist_tools)

        # Agent 2 Tools: TOC & Page OCR
        self.reader_tools = [
            inspect_table_of_contents,
            read_page_ocr,
        ]
        self.reader_tools_map = {t.name: t for t in self.reader_tools}
        self.reader_llm = self.llm.bind_tools(self.reader_tools)

        # LangGraph StateGraph
        self.workflow = None
        if LANGGRAPH_STATE_AVAILABLE:
            self._build_state_graph()

    def _run_subagent_loop(
        self,
        agent_label: str,
        bound_llm,
        tools_map: Dict[str, Any],
        system_prompt: str,
        task_input: str,
        step_budget: int,
    ) -> tuple[str, List[Any]]:
        """Runs a bounded tool-calling loop for a single specialized agent."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task_input),
        ]
        steps = []

        for _ in range(step_budget):
            response = bound_llm.invoke(messages)
            messages.append(response)

            if not getattr(response, "tool_calls", None):
                return response.content, steps

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args", {})
                tool_id = tc.get("id", tool_name)

                tool_obj = tools_map.get(tool_name)
                if tool_obj:
                    try:
                        observation = tool_obj.invoke(tool_args)
                    except Exception as err:
                        observation = f"Error executing '{tool_name}': {err}"
                else:
                    observation = f"Error: Tool '{tool_name}' is not assigned to {agent_label}."

                # Tag tool name with the agent role for clear trace visibility
                tagged_tool = f"[{agent_label}] {tool_name}"
                action = AgentAction(tool=tagged_tool, tool_input=tool_args, log=str(tc))
                steps.append((action, observation))

                messages.append(
                    ToolMessage(
                        content=str(observation),
                        tool_call_id=tool_id,
                    )
                )

        messages.append(HumanMessage(content="Summarize your findings now based on the tool observations above."))
        final_resp = self.llm.invoke(messages)
        return final_resp.content, steps

    def _archivist_node(self, state: MultiAgentState) -> Dict[str, Any]:
        budget = max(3, self.max_steps // 2)
        notes, steps = self._run_subagent_loop(
            agent_label="Agent1:Archivist",
            bound_llm=self.archivist_llm,
            tools_map=self.archivist_tools_map,
            system_prompt=ARCHIVIST_PROMPT,
            task_input=f"User Research Question: {state['user_query']}",
            step_budget=budget,
        )
        return {
            "archivist_notes": notes,
            "intermediate_steps": state.get("intermediate_steps", []) + steps,
        }

    def _reader_node(self, state: MultiAgentState) -> Dict[str, Any]:
        budget = max(4, self.max_steps // 2)
        reader_input = (
            f"Original User Question: {state['user_query']}\n\n"
            f"--- Handoff from Agent 1 (Archivist) ---\n{state['archivist_notes']}\n\n"
            f"Now inspect the Table of Contents and read the relevant page(s)."
        )
        notes, steps = self._run_subagent_loop(
            agent_label="Agent2:PageReader",
            bound_llm=self.reader_llm,
            tools_map=self.reader_tools_map,
            system_prompt=READER_PROMPT,
            task_input=reader_input,
            step_budget=budget,
        )
        return {
            "reader_notes": notes,
            "intermediate_steps": state.get("intermediate_steps", []) + steps,
        }

    def _synthesizer_node(self, state: MultiAgentState) -> Dict[str, Any]:
        synth_input = (
            f"Original User Question: {state['user_query']}\n\n"
            f"=== Report from Agent 1 (Catalog & Terminology Archivist) ===\n"
            f"{state['archivist_notes']}\n\n"
            f"=== Report from Agent 2 (TOC & Primary Page Reader) ===\n"
            f"{state['reader_notes']}\n\n"
            f"Write the final grounded response with full provenance (Document ID, Section, Page, Quote), "
            f"or issue an explicit Refusal if the evidence does not answer the question."
        )
        messages = [
            SystemMessage(content=SYNTHESIZER_PROMPT),
            HumanMessage(content=synth_input),
        ]
        response = self.llm.invoke(messages)

        # Agent 3 synthesis handoff in trace
        synth_action = AgentAction(
            tool="[Agent3:Synthesizer] verify_and_synthesize",
            tool_input={"sources": ["Archivist Report", "PageReader Excerpts"]},
            log="Synthesis & Grounding Verification",
        )
        steps = state.get("intermediate_steps", []) + [(synth_action, "Verified evidence & generated synthesis.")]
        return {
            "final_output": response.content,
            "intermediate_steps": steps,
        }

    def _build_state_graph(self):
        builder = StateGraph(MultiAgentState)
        builder.add_node("archivist", self._archivist_node)
        builder.add_node("page_reader", self._reader_node)
        builder.add_node("synthesizer", self._synthesizer_node)

        builder.add_edge(START, "archivist")
        builder.add_edge("archivist", "page_reader")
        builder.add_edge("page_reader", "synthesizer")
        builder.add_edge("synthesizer", END)

        self.workflow = builder.compile()

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_query = inputs.get("input", "")
        initial_state: MultiAgentState = {
            "user_query": user_query,
            "archivist_notes": "",
            "reader_notes": "",
            "final_output": "",
            "intermediate_steps": [],
        }

        if self.workflow is not None:
            final_state = self.workflow.invoke(initial_state)
        else:
            s1 = self._archivist_node(initial_state)
            initial_state.update(s1)
            s2 = self._reader_node(initial_state)
            initial_state.update(s2)
            s3 = self._synthesizer_node(initial_state)
            initial_state.update(s3)
            final_state = initial_state

        return {
            "output": final_state.get("final_output", "No response generated."),
            "intermediate_steps": final_state.get("intermediate_steps", []),
            "archivist_notes": final_state.get("archivist_notes", ""),
            "reader_notes": final_state.get("reader_notes", ""),
        }


def create_dissify_multi_agent(
    model_name: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    max_steps: int = MAX_AGENT_STEPS,
    num_ctx: int = OLLAMA_NUM_CTX,
    num_predict: int = OLLAMA_NUM_PREDICT,
):
    """Initializes the 3-Agent Multi-Agent Orchestrator."""
    return DissifyMultiAgentExecutor(
        model_name=model_name,
        base_url=base_url,
        max_steps=max_steps,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )
