"""The crew: connects the YAML config with the LLM and the tools."""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, llm, task, tool

from scripts.llm import local_llm
from tools.list_cards import ListCardsTool
from tools.read_card import ReadCardTool
from tools.search_cards import SearchCardsTool


@CrewBase
class StarterCrew:
    """Two agents that run one after the other for every question.

    1. card_guide answers the question (task: answer_question).
    2. suggestion_helper suggests follow-up questions (task: suggest_questions).

    Naming rules:
    - Each @agent / @task method name must match its key in agents.yaml /
      tasks.yaml, and each @tool method name must match the names listed
      under "tools:" in agents.yaml.
    - The tasks run in the order their @task methods are defined here.
    """

    agents_config = "../config/agents.yaml"
    tasks_config = "../config/tasks.yaml"

    @llm
    def local(self):
        return local_llm()

    @tool
    def list_cards(self):
        return ListCardsTool()

    @tool
    def search_cards(self):
        return SearchCardsTool()

    @tool
    def read_card(self):
        return ReadCardTool()

    @agent
    def card_guide(self) -> Agent:
        return Agent(config=self.agents_config["card_guide"])

    @agent
    def suggestion_helper(self) -> Agent:
        return Agent(config=self.agents_config["suggestion_helper"])

    @task
    def answer_question(self) -> Task:
        return Task(config=self.tasks_config["answer_question"])

    @task
    def suggest_questions(self) -> Task:
        return Task(config=self.tasks_config["suggest_questions"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
