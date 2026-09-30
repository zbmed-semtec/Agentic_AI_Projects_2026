"""When a guardrail fails, recall the same agent, then let the crew continue."""

from collections.abc import Callable

from crewai import Agent, Crew, Task
from crewai.tasks.llm_guardrail import LLMGuardrail
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.formatter import aggregate_raw_outputs_from_tasks


def use_continuing_guardrails(crew: Crew) -> Crew:
    """Replace each task's guardrail so a failure recalls that task's agent.

    CrewAI's own retry stops the crew after the last failed check. This wrapper
    recalls the agent with the guardrail feedback and returns the revision, so
    the next task still runs.
    """
    for task in crew.tasks:
        description = _guardrail_text(task)
        agent = task.agent
        if not description or not isinstance(agent, Agent) or agent.llm is None:
            continue
        checker = LLMGuardrail(description=description, llm=agent.llm)
        wrapped = _make_guardrail(task, agent, checker)
        task.guardrail = wrapped
        task._guardrail = wrapped
    return crew


def _guardrail_text(task: Task) -> str | None:
    if isinstance(task.guardrail, str):
        return task.guardrail
    inner = getattr(task, "_guardrail", None)
    description = getattr(inner, "description", None)
    return description if isinstance(description, str) else None


def _make_guardrail(
    task: Task,
    agent: Agent,
    checker: LLMGuardrail,
) -> Callable[[TaskOutput], tuple[bool, str]]:
    recalls = task.guardrail_max_retries

    def guardrail(task_output: TaskOutput) -> tuple[bool, str]:
        passed, feedback = checker(task_output)
        if passed:
            return True, task_output.raw

        current = task_output.raw
        note = str(feedback)
        for attempt in range(1, recalls + 1):
            label = task.name or "task"
            print(
                f"\nGuardrail failed on {label}. "
                f"Recalling {agent.role} to correct ({attempt}/{recalls}).\n"
            )
            revised = agent.execute_task(
                task=task,
                context=_correction_context(task, current, note),
            )
            current = revised if isinstance(revised, str) else str(revised)
            passed, feedback = checker(task_output.model_copy(update={"raw": current}))
            if passed:
                print("Correction accepted. Continuing to the next step.\n")
                return True, current
            note = str(feedback)

        label = task.name or "task"
        print(
            f"Guardrail still flags {label}. "
            "Continuing with the latest correction.\n"
        )
        return True, current

    return guardrail


def _correction_context(task: Task, previous: str, feedback: str) -> str:
    earlier = ""
    if isinstance(task.context, list):
        earlier = aggregate_raw_outputs_from_tasks(task.context)
    parts = [part for part in (earlier, "") if part]
    parts.append(
        "### Previous attempt failed validation:\n"
        f"{feedback}\n\n"
        "### Previous result:\n"
        f"{previous}\n\n"
        "Revise the previous result so it passes the check. "
        "Keep every part that already passed."
    )
    return "\n\n".join(parts)
