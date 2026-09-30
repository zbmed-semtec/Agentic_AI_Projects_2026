"""Same four-step literature triage, owned by one agent."""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, llm, task

from starter.continuing_guardrail import use_continuing_guardrails
from starter.llm import local_llm


@CrewBase
class SingleAgentCrew:
    """One analyst runs screen, extract, compare, then write."""

    agents_config = "config/single_agent.yaml"
    tasks_config = "config/single_agent_tasks.yaml"

    @llm
    def local(self):
        return local_llm()

    @agent
    def triage_agent(self) -> Agent:
        return Agent(config=self.agents_config["triage_agent"])

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
    def synthesis_task(self) -> Task:
        return Task(config=self.tasks_config["synthesis_task"])

    @crew
    def crew(self) -> Crew:
        built = Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
        return use_continuing_guardrails(built)
