"""Crew definition. Add agents and tasks here, and describe them in config/."""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, llm, task, tool

from starter.continuing_guardrail import use_continuing_guardrails
from starter.llm import local_llm
from starter.router import steps_for
from starter.tools.example_tool import WordCountTool


@CrewBase
class StarterCrew:
    """Literature triage. The route decides which agents run after screening."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    route = "evidence"

    @llm
    def local(self):
        return local_llm()

    @tool
    def word_count(self):
        return WordCountTool()

    @agent
    def relevance_agent(self) -> Agent:
        return Agent(config=self.agents_config["relevance_agent"])

    @agent
    def extraction_agent(self) -> Agent:
        return Agent(config=self.agents_config["extraction_agent"])

    @agent
    def contradiction_agent(self) -> Agent:
        return Agent(config=self.agents_config["contradiction_agent"])

    @agent
    def coverage_agent(self) -> Agent:
        return Agent(config=self.agents_config["coverage_agent"])

    @agent
    def synthesis_agent(self) -> Agent:
        return Agent(config=self.agents_config["synthesis_agent"])

    @task
    def relevance_task(self) -> Task:
        return Task(config=self.tasks_config["relevance_task"])

    @task
    def extraction_task(self) -> Task:
        return Task(config=self.tasks_config["extraction_task"])

    @task
    def contradiction_task(self) -> Task:
        return Task(config=self.tasks_config["contradiction_task"])

    @task
    def coverage_task(self) -> Task:
        return Task(config=self.tasks_config["coverage_task"])

    @task
    def synthesis_task(self) -> Task:
        return Task(config=self.tasks_config["synthesis_task"])

    def _tasks_for_route(self) -> list[Task]:
        steps = steps_for(self.route)
        by_name = {
            "extraction": self.extraction_task(),
            "contradiction": self.contradiction_task(),
            "coverage": self.coverage_task(),
        }
        selected = [self.relevance_task()]
        selected.extend(by_name[name] for name in steps)
        synthesis = self.synthesis_task()
        synthesis.context = list(selected)
        selected.append(synthesis)
        return selected

    @crew
    def crew(self) -> Crew:
        tasks = self._tasks_for_route()
        agents = []
        seen: set[int] = set()
        for item in tasks:
            if item.agent is None or id(item.agent) in seen:
                continue
            seen.add(id(item.agent))
            agents.append(item.agent)
        built = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )
        return use_continuing_guardrails(built)
