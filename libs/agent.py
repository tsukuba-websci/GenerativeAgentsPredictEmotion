import sys

sys.path.append(".")
import random
from typing import Any, List

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from pydantic import BaseModel

from libs.memory import CaptioningAgentMemory


class CaptioningAgent(BaseModel):
    """An Agent as a character with memory and innate characteristics."""

    name: str
    """The character's name."""
    situation: str = ""
    """The situation that the agent is in."""
    memory: CaptioningAgentMemory
    """The memory object that combines relevance, recency, and 'importance'."""
    llm: BaseLanguageModel
    """The underlying language model."""
    llm_panas: BaseLanguageModel
    """The underlying language model for PANAS Score."""
    verbose: bool = False
    """How frequently to re-generate the summary."""
    contextual_understanding_template: Any = PromptTemplate.from_template(
        "Below is the background context of your memories:\n"
        "---\n"
        "{norm}\n"
        "---\n"
        "The following is the description of the most recent memory: \n"
        "---\n"
        "{memory}\n"
        "---\n"
        "Given the background context of the memories, how can we interpret the new situation?"
    )
    panas_test_template: str = PromptTemplate.from_template(
        "You can only reply the numbers from 1 to 5.\n"
        "{scenario}\n"
        "Please indicate the extent of your feeling in all the following emotions on a scale of 1 to 5.\n"
        '1 denotes "very slightly or not at all", 2 denotes "a little", 3 denotes "moderately", 4 denotes "quite a bit", and 5 denotes "extremely".\n'
        "Please score all emotions one by one using the scale from 1 to 5:"
        "{emotions}"
        "\n"
        "Generally, your score should not be all ones. Your answer should be realistic and reasonable.\n"
    )

    def __init__(
        self,
        name: str,
        saliency_weight: float,
        reinforcement_weight: float,
        recency_weight: float,
        relevance_weight: float,
        situation: str = "",
        verbose: bool = False,
        delete: bool = False,
    ):
        """Initialize the agent."""
        llm_instance = ChatOpenAI(
            model_name="gpt-4-1106-preview",
            max_tokens=1000,
            request_timeout=120,
            max_retries=20,
        )
        llm_instance_panas = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            max_tokens=1000,
            request_timeout=120,
            max_retries=20,
            temperature=0,
        )
        memory_instance = CaptioningAgentMemory(
            llm=llm_instance,
            saliency_weight=saliency_weight,
            reinforcement_weight=reinforcement_weight,
            recency_weight=recency_weight,
            relevance_weight=relevance_weight,
            verbose=verbose,
            experiment_id=name,
            delete=delete,
        )

        super().__init__(
            name=name,
            situation=situation,
            memory=memory_instance,
            llm=llm_instance,
            llm_panas=llm_instance_panas,
            verbose=verbose,
        )

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    def chain_panas(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm_panas, prompt=prompt, verbose=self.verbose)

    def shuffle_emotions(self) -> str:
        emotions_list = [
            "Attentive",
            "Active",
            "Alert",
            "Excited",
            "Enthusiastic",
            "Determined",
            "Inspired",
            "Proud",
            "Interested",
            "Strong",
            "Hostile",
            "Irritable",
            "Ashamed",
            "Guilty",
            "Distressed",
            "Upset",
            "Scared",
            "Afraid",
            "Jittery",
            "Nervous",
        ]
        random.shuffle(emotions_list)

        emotions_string = ""
        for emotion in emotions_list:
            emotions_string += f"{emotion}:\n"

        return emotions_string

    def check_panas_format(self, panas_str: str) -> bool:
        # Split the file content into lines and strip any whitespace
        lines = [line.strip() for line in panas_str.split("\n") if line.strip()]

        # Check if there are exactly 20 lines
        if len(lines) != 20:
            return False

        # Check each line for the correct format: "Emotion: Number"
        for line in lines:
            parts = line.split(":")
            if len(parts) != 2:
                return False

            emotion, number = parts
            try:
                number = int(number.strip())
                if number < 1 or number > 5:
                    return False
            except ValueError:
                return False

        return True

    def run_text(self, memories: List[str]):
        """
        Run the architecture on a text-based memories.

        Args:
            images: A list of str, representing the memories of the agent.

        Returns:
            None
        """

        for index, memory in enumerate(memories, start=1):
            memory_embedding = self.memory.embeddings_model.embed_query(memory)

            self.memory.create_norm(memory_embedding)

            contextual_understanding = self.chain(prompt=self.contextual_understanding_template).run(
                norm=self.memory.norm, memory=memory
            )

            panas_scores = ""
            while not self.check_panas_format(panas_scores):
                panas_scores = self.chain_panas(prompt=self.panas_test_template).run(
                    scenario=contextual_understanding, emotions=self.shuffle_emotions()
                )

            saliency_score = self.memory.calculate_saliency_score(memory)

            self.memory.add_nodes_and_edges(
                panas_scores=panas_scores,
                memory=memory,
                memory_embedding=memory_embedding,
                memory_number=index,
                situation=self.situation,
                saliency_score=saliency_score,
                contextual_understanding=contextual_understanding,
            )

    def run_text_no_retrieval(self, memories: List[str]):
        """
        Run the architecture on a text-based memories, without memory retrieval.

        Args:
            images: A list of str, representing the memories of the agent.

        Returns:
            None
        """

        for index, memory in enumerate(memories, start=1):
            self.memory.create_norm_no_retrieval()

            panas_scores = ""
            count = 0
            while not self.check_panas_format(panas_scores):
                contextual_understanding = self.chain(prompt=self.contextual_understanding_template).run(
                    norm=self.memory.norm, memory=memory
                )
                panas_scores = self.chain_panas(prompt=self.panas_test_template).run(
                    scenario=contextual_understanding, emotions=self.shuffle_emotions()
                )
                count += 1
                if count > 30:
                    break

            self.memory.add_nodes_and_edges(
                panas_scores=panas_scores,
                memory=memory,
                memory_number=index,
                situation=self.situation,
                contextual_understanding=contextual_understanding,
            )

    def run_text_no_norm(self, memories: List[str]):
        """
        Run the architecture on a text-based memories, without the norm.

        Args:
            images: A list of str, representing the memories of the agent.

        Returns:
            None
        """

        for index, memory in enumerate(memories, start=1):
            panas_scores = ""
            while not self.check_panas_format(panas_scores):
                panas_scores = self.chain_panas(prompt=self.panas_test_template).run(
                    scenario=memory, emotions=self.shuffle_emotions()
                )

            self.memory.add_nodes_and_edges(
                panas_scores=panas_scores,
                memory=memory,
                memory_number=index,
                situation=self.situation,
            )
