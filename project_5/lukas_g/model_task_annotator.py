"""Crew for evidence-based annotation of missing model-task labels."""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, llm, task, tool

from starter.llm import local_llm
from starter.tools.hf_task_taxonomy import ListHuggingFaceTasksTool
from starter.tools.model_data_tools import ReadSampledModelsTool, WriteModelMLTaskTool
from starter.tools.publication_tools import ReadPublicationPDFTool, SearchPublicationsTool


@CrewBase
class ModelTaskAnnotatorCrew:
    """Annotate missing mlTask values without changing uncertain records."""

    agents_config = "config/annotation_agents.yaml"
    tasks_config = "config/annotation_tasks.yaml"

    @llm
    def local(self):
        return local_llm()

    @tool
    def read_sampled_models(self):
        return ReadSampledModelsTool()

    @tool
    def list_huggingface_tasks(self):
        return ListHuggingFaceTasksTool()

    @tool
    def write_model_mltask(self):
        return WriteModelMLTaskTool()

    @tool
    def search_publications(self):
        return SearchPublicationsTool()

    @tool
    def read_publication_pdf(self):
        return ReadPublicationPDFTool()

    @agent
    def model_task_annotator(self) -> Agent:
        return Agent(config=self.agents_config["model_task_annotator"])

    @task
    def annotate_model_tasks(self) -> Task:
        return Task(config=self.tasks_config["annotate_model_tasks"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
