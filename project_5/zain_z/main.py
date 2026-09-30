"""Extract properties of one Hugging Face model into output/<model>.json.

    python main.py microsoft/phi-1_5            # all properties
    python main.py microsoft/phi-1_5 mlTask     # only the given ones

"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from crew import PROPERTIES, Finding, extract

HERE = Path(__file__).parent
OUTPUT = HERE / "output"
LOGS = HERE / ".log"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


class Tee:
    """Write to the terminal and to a log file (without color codes)."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(ANSI.sub("", text))
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_log() -> Path:
    """Copy everything printed from here on into .log/<timestamp>.log."""
    LOGS.mkdir(exist_ok=True)
    path = LOGS / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
    log = open(path, "w", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    return path


def to_entry(finding: Finding) -> dict:
    """Turn a Finding into the ground-truth shape: value, status, evidence, notes."""
    if not finding.value:
        return {"value": None, "status": "unknown", "evidence": [], "notes": ""}
    return {
        "value": finding.value,
        "status": "found" if finding.confident else "review",
        "evidence": [e.model_dump() for e in finding.evidence],
        "notes": "",
    }


def main() -> None:
    log_path = start_log()
    print(f"Run: {' '.join(sys.argv)}")
    load_dotenv(HERE / ".env")
    model_id = sys.argv[1] if len(sys.argv) > 1 else "microsoft/phi-1_5"
    props = sys.argv[2:] or list(PROPERTIES)

    path = OUTPUT / f"{model_id.replace('/', '__')}.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {model_id: {}}

    for prop in props:
        if data[model_id].get(prop, {}).get("value") is not None:
            print(f"\n{prop}: already in {path.name}, skipping")
            continue
        data[model_id][prop] = to_entry(extract(model_id, prop))
        print(f"\n{prop}: {json.dumps(data[model_id][prop], ensure_ascii=False)}")
        OUTPUT.mkdir(exist_ok=True)
        path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
