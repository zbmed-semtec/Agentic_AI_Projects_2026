"""
Logging Service for Dissify Exploration Agent.
Tracks user queries, complete AI responses, Think-Act-Observe tool steps,
and execution metadata into structured JSON files and a JSONL stream.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from config import LOGS_DIR


class QueryLogger:
    """Handles persistent logging of queries, intermediate reasoning steps, and agent responses."""

    def __init__(self, logs_dir: Path = LOGS_DIR):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.summary_file = self.logs_dir / "queries_summary.jsonl"

    def log_session(
        self,
        user_query: str,
        final_response: str,
        intermediate_steps: List[Any],
        model_name: str,
        execution_time_sec: float,
        architecture: str = "single_agent",
        extra_metadata: Dict[str, Any] = None,
        error: str = None
    ) -> Path:
        """
        Saves a complete record of the query execution to a dedicated JSON file
        and appends a compact summary to queries_summary.jsonl.
        """
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        query_id = str(uuid.uuid4())[:8]

        # Extract structured tool steps
        formatted_steps = []
        tools_called = []
        for idx, (action, observation) in enumerate(intermediate_steps, 1):
            tool_name = getattr(action, "tool", str(action))
            tool_args = getattr(action, "tool_input", {})
            tools_called.append(tool_name)

            formatted_steps.append({
                "step": idx,
                "tool": tool_name,
                "input": tool_args,
                "observation": str(observation)
            })

        # Determine response status
        if error:
            status = "ERROR"
        elif "refusal" in final_response.lower():
            status = "REFUSAL"
        elif "maximum reasoning steps reached" in final_response.lower() or "need more steps" in final_response.lower():
            status = "STEP_LIMIT_REACHED"
        else:
            status = "SUCCESS_GROUNDED"

        log_data = {
            "query_id": query_id,
            "timestamp": now.isoformat(),
            "architecture": architecture,
            "model_name": model_name,
            "status": status,
            "execution_time_sec": round(execution_time_sec, 2),
            "user_query": user_query,
            "total_steps": len(formatted_steps),
            "tools_called": tools_called,
            "final_response": final_response,
            "intermediate_steps": formatted_steps,
            "extra_metadata": extra_metadata or {},
            "error": error
        }

        # 1. Save detailed query log
        log_filename = f"query_{timestamp_str}_{query_id}.json"
        log_path = self.logs_dir / log_filename
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Logging Warning] Failed to write detailed log: {e}")

        # 2. Append to rolling summary JSONL
        try:
            summary_entry = {
                "query_id": query_id,
                "timestamp": now.isoformat(),
                "architecture": architecture,
                "model_name": model_name,
                "status": status,
                "execution_time_sec": round(execution_time_sec, 2),
                "total_steps": len(formatted_steps),
                "tools_called": list(set(tools_called)),
                "user_query": user_query,
                "log_file": str(log_path.name)
            }
            with open(self.summary_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Logging Warning] Failed to update summary log: {e}")

        return log_path
