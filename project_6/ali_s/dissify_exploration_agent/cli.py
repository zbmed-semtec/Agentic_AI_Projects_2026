"""
Interactive CLI for Dissify Exploration Agent.
Supports Single-Agent ReAct, 3-Agent Multi-Agent Orchestration, and Side-by-Side Comparison mode.
Features rich terminal rendering of Think-Act-Observe loops, citations, and refusal states.
"""

import sys
import argparse
import time
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt

from config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    MAX_AGENT_STEPS,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
)
from data_loader import DissifyDataManager
from agent import create_dissify_agent
from multi_agent import create_dissify_multi_agent
from logger_service import QueryLogger

console = Console()
query_logger = QueryLogger()

SAMPLE_QUERIES = [
    "What chemical methods did Franz Prachfeld use to measure milk fat content, and on which pages are the test results documented?",
    "Which theses investigate dairy milk composition, what historical terms were used for lactation, and what are the author's vital dates for top 3 findings?",
    "What did dissertation 10013639 report about antibiotics in 1905?", #(Refusal Test)
]


def print_banner(data_mgr: DissifyDataManager, model_name: str, mode: str, max_steps: int, recursion_limit: int):
    banner = Text()
    banner.append("🏛️  DISSIFY EXPLORATION AGENT\n", style="bold cyan")
    banner.append("Autonomous Agentic AI for Historical Dissertations (1880–1950)\n", style="dim white")
    banner.append(
        f"Model: {model_name} (Ollama) | Mode: {mode.upper()} | Max Steps: {max_steps} | Recursion Limit: {recursion_limit}\n",
        style="italic yellow",
    )
    banner.append(
        f"Corpus Loaded: {len(data_mgr.documents)} theses | {len(data_mgr.terminology_index)} terminology pairs | {len(data_mgr.cv_timelines)} author profiles",
        style="green",
    )

    console.print(Panel(banner, border_style="cyan", padding=(1, 2)))


def run_single_execution(
    agent_executor,
    query: str,
    architecture: str,
    model_name: str = OLLAMA_MODEL,
) -> Dict[str, Any]:
    arch_title = "SINGLE-AGENT (ReAct)" if architecture == "single_agent" else "MULTI-AGENT (Archivist → Reader → Synthesizer)"
    console.print(f"\n[bold cyan]━━━ Running {arch_title} ━━━[/bold cyan]")
    console.print(f"[bold yellow]🔍 Query:[/bold yellow] [white]{query}[/white]\n")

    start_time = time.time()
    steps = []
    error_msg = None
    extra_meta = {}

    with console.status(f"[bold green]{arch_title} executing...[/bold green]", spinner="dots"):
        try:
            result = agent_executor.invoke({"input": query})
            output = result.get("output", "No response generated.")
            steps = result.get("intermediate_steps", [])
            if architecture == "multi_agent":
                extra_meta = {
                    "archivist_notes": result.get("archivist_notes", ""),
                    "reader_notes": result.get("reader_notes", ""),
                }
        except Exception as e:
            error_msg = str(e)
            output = f"Execution error: {error_msg}"
            console.print(f"[bold red]Execution error:[/bold red] {e}")

    execution_time = time.time() - start_time

    if steps:
        table = Table(
            title=f"{arch_title} Execution Trace (Think → Act → Observe)",
            border_style="dim blue",
            show_lines=True,
        )
        table.add_column("Step", style="bold cyan", width=6)
        table.add_column("Tool Action / Agent Role", style="bold yellow", width=32)
        table.add_column("Input Arguments", style="white", width=28)
        table.add_column("Observation Summary", style="dim white")

        for idx, (action, observation) in enumerate(steps, 1):
            obs_preview = str(observation).strip().replace("\n", " ")
            if len(obs_preview) > 110:
                obs_preview = obs_preview[:110] + "..."
            table.add_row(
                str(idx),
                action.tool,
                str(action.tool_input),
                obs_preview,
            )
        console.print(table)

    # Highlight Refusals vs Step Limit vs Grounded Answers
    if "refusal" in output.lower():
        style = "bold magenta"
        title = f"🚫 [{arch_title}] Agent Refusal (Evidence Missing / Not In Corpus)"
        status = "REFUSAL"
    elif "need more steps" in output.lower() or "maximum reasoning steps reached" in output.lower():
        style = "bold yellow"
        title = f"⚠️ [{arch_title}] Step Limit Reached"
        status = "STEP_LIMIT"
    elif error_msg:
        style = "bold red"
        title = f"❌ [{arch_title}] Execution Error"
        status = "ERROR"
    else:
        style = "bold green"
        title = f"📄 [{arch_title}] Grounded Research Synthesis"
        status = "SUCCESS_GROUNDED"

    console.print(Panel(output, title=title, border_style=style, padding=(1, 2)))

    # Persist complete trace and metadata into logs folder
    log_file = query_logger.log_session(
        user_query=query,
        final_response=output,
        intermediate_steps=steps,
        model_name=model_name,
        execution_time_sec=execution_time,
        architecture=architecture,
        extra_metadata=extra_meta,
        error=error_msg,
    )
    console.print(
        f"[dim]💾 Logged ({architecture}) to: {log_file.relative_to(log_file.parent.parent)} (took {execution_time:.1f}s)[/dim]\n"
    )

    return {
        "architecture": architecture,
        "execution_time": execution_time,
        "steps_count": len(steps),
        "tools_used": [getattr(a, "tool", str(a)) for a, _ in steps],
        "status": status,
        "output_length": len(output),
        "log_file": str(log_file.name),
    }


def print_comparison_summary(single_stats: Dict[str, Any], multi_stats: Dict[str, Any]):
    comp_table = Table(
        title="⚡ Single-Agent vs. Multi-Agent Benchmark Comparison",
        border_style="bold yellow",
        show_lines=True,
    )
    comp_table.add_column("Metric", style="bold white", width=24)
    comp_table.add_column("Single-Agent (ReAct)", style="cyan", width=36)
    comp_table.add_column("Multi-Agent (3-Agent Pipeline)", style="green", width=36)

    comp_table.add_row(
        "Execution Latency",
        f"{single_stats['execution_time']:.2f} sec",
        f"{multi_stats['execution_time']:.2f} sec",
    )
    comp_table.add_row(
        "Total Steps / Actions",
        str(single_stats["steps_count"]),
        str(multi_stats["steps_count"]),
    )
    comp_table.add_row(
        "Resolution Status",
        single_stats["status"],
        multi_stats["status"],
    )
    comp_table.add_row(
        "Synthesis Length",
        f"{single_stats['output_length']} chars",
        f"{multi_stats['output_length']} chars",
    )
    comp_table.add_row(
        "Tools Sequence",
        " → ".join(single_stats["tools_used"]) or "None",
        " → ".join(multi_stats["tools_used"]) or "None",
    )
    comp_table.add_row(
        "Saved Log File",
        single_stats["log_file"],
        multi_stats["log_file"],
    )

    console.print(comp_table)


def dispatch_query(mode: str, single_agent, multi_agent, query: str, model_name: str):
    if mode == "single":
        run_single_execution(single_agent, query, "single_agent", model_name=model_name)
    elif mode == "multi":
        run_single_execution(multi_agent, query, "multi_agent", model_name=model_name)
    elif mode == "compare":
        s_stats = run_single_execution(single_agent, query, "single_agent", model_name=model_name)
        m_stats = run_single_execution(multi_agent, query, "multi_agent", model_name=model_name)
        print_comparison_summary(s_stats, m_stats)


def main():
    parser = argparse.ArgumentParser(description="Dissify Exploration Agent CLI")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help=f"Ollama model name (default: {OLLAMA_MODEL})")
    parser.add_argument("--url", type=str, default=OLLAMA_BASE_URL, help="Ollama API base URL")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multi", "compare"],
        default="single",
        help="Execution mode: 'single' (ReAct), 'multi' (3-agent pipeline), or 'compare' (run both side-by-side)",
    )
    parser.add_argument("--max-steps", type=int, default=MAX_AGENT_STEPS, help=f"Maximum tool execution steps (default: {MAX_AGENT_STEPS})")
    parser.add_argument("--recursion-limit", type=int, default=None, help="LangGraph recursion limit (default: max(50, max_steps * 4))")
    parser.add_argument("--num-ctx", type=int, default=OLLAMA_NUM_CTX, help=f"Ollama context window tokens (default: {OLLAMA_NUM_CTX})")
    parser.add_argument("--num-predict", type=int, default=OLLAMA_NUM_PREDICT, help=f"Maximum output tokens to generate (default: {OLLAMA_NUM_PREDICT})")
    parser.add_argument("--query", type=str, default=None, help="Run single query and exit")
    args = parser.parse_args()

    calc_recursion = args.recursion_limit or max(50, args.max_steps * 4)
    current_mode = args.mode

    console.print("[dim]Initializing data manager and indexing corpus...[/dim]")
    data_mgr = DissifyDataManager.get_instance()
    print_banner(data_mgr, args.model, current_mode, args.max_steps, calc_recursion)

    console.print(f"[dim]Connecting to local model '{args.model}' at {args.url}...[/dim]")
    try:
        single_agent = create_dissify_agent(
            model_name=args.model,
            base_url=args.url,
            max_steps=args.max_steps,
            recursion_limit=args.recursion_limit,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
        )
        multi_agent = create_dissify_multi_agent(
            model_name=args.model,
            base_url=args.url,
            max_steps=args.max_steps,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
        )
    except Exception as e:
        console.print(f"[bold red]Failed to initialize agents:[/bold red] {e}")
        console.print("[yellow]Make sure Ollama is running (`ollama serve`).[/yellow]")
        sys.exit(1)

    if args.query:
        dispatch_query(current_mode, single_agent, multi_agent, args.query, args.model)
        return

    console.print("\n[bold]Quick Commands:[/bold]")
    console.print("  Run sample queries:                                [cyan]1, 2, 3[/cyan]")
    console.print("  Switch to Single-Agent ReAct mode:                 [cyan]mode single[/cyan]")
    console.print("  Switch to Multi-Agent pipeline:                    [cyan]mode multi[/cyan]")
    console.print("  Run BOTH architectures side-by-side and compare!:  [cyan]mode compare[/cyan]")
    console.print("  Quit application:                                  [magenta]exit / quit(q)[/magenta]\n")

    while True:
        try:
            user_input = Prompt.ask(f"[bold cyan]({current_mode.upper()}) Enter query[/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[dim]Exiting Dissify Exploration Agent. Goodbye![/dim]")
                break
            if user_input.lower() in ("mode single", "mode multi", "mode compare"):
                current_mode = user_input.lower().split()[1]
                console.print(f"[bold green]✓ Switched execution mode to: {current_mode.upper()}[/bold green]\n")
                continue
            if user_input in ("1", "2", "3"):
                selected_query = SAMPLE_QUERIES[int(user_input) - 1]
                dispatch_query(current_mode, single_agent, multi_agent, selected_query, args.model)
            else:
                dispatch_query(current_mode, single_agent, multi_agent, user_input, args.model)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session terminated.[/dim]")
            break


if __name__ == "__main__":
    main()
