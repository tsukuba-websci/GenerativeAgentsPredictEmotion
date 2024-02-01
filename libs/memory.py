from datetime import datetime
import os
import re
from typing import List
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.embeddings.openai import OpenAIEmbeddings
from neo4j import GraphDatabase, Driver
from pydantic import BaseModel, Field
from typing import Any


def neo4j_date_to_datetime(dt: Any) -> datetime:
    pdt = datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, int(dt.second), int(dt.second * 1000000 % 1000000))
    return pdt


def _get_days_passed(time: datetime, ref_time: datetime) -> float:
    """Get the days passed between two datetime objects."""
    return (time - ref_time).total_seconds() / 86400


class CaptioningAgentMemory(BaseModel):
    """Memory for the generative agent."""

    llm: BaseLanguageModel
    """The underlying language model."""
    embeddings_model: OpenAIEmbeddings
    """The underlying embeddings model."""
    verbose: bool = False
    """To stream the output of the language model or not."""
    neo4j_driver: Driver
    """The neo4j driver."""
    experiment_id: str
    """The id of the current experiment."""
    norm: str = ""
    """The current norm, or background context, of the agent."""
    norm_id: int = 0
    """The id of the current norm node."""
    current_norm_ids: List[str] = []
    """The ids of the memory nodes which created the current norm."""
    current_memory_id: int = None
    """The id of the current memory node."""
    memories_in_norm: int = 5
    """How many memories to include in the norm."""

    # parameters of the memory retriever
    saliency_weight: float
    """The weight of the saliency score."""
    reinforcement_weight: float
    """The weight of the reinforcement score."""
    recency_weight: float
    """The weight of the recency score."""
    relevance_weight: float
    """The weight of the relevance score."""

    decay_rate: float = Field(default=0.01)
    """The exponential decay factor used as (1.0-decay_rate)**(days_passed)."""
    rho: int = 1
    """The reinforcement increment."""
    memories_to_reinforce: int = 5
    """The number of memories to reinforce."""

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        llm: BaseLanguageModel,
        saliency_weight: float,
        reinforcement_weight: float,
        recency_weight: float,
        relevance_weight: float,
        experiment_id: str,
        verbose: bool = False,
        delete: bool = False,
    ):
        """Initialize the memory."""
        # Setup neo4j
        NEO4J_URI = os.getenv("NEO4J_URI")
        NEO4J_USER = os.getenv("NEO4J_USER")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        with neo4j_driver.session() as session:
            # Delete all nodes and relationships

            if delete:
                print(f"Deleting all nodes with experiment_id: {experiment_id}")
                session.run(
                    """
                                MATCH (n {
                                    experiment_id: $experiment_id
                                })
                                DETACH DELETE n;
                            """,
                    experiment_id=experiment_id,
                )

            # Delete vector index if exists
            session.run(
                """
                DROP INDEX `memory_embeddings` IF EXISTS;
            """
            )

            # Create vector index
            session.run(
                """
                CALL db.index.vector.createNodeIndex(
                    'memory_embeddings',
                    'Memory',
                    'memory_embedding',
                    1536,
                    'cosine'
                )
            """
            )

        super().__init__(
            llm=llm,
            verbose=verbose,
            neo4j_driver=neo4j_driver,
            embeddings_model=OpenAIEmbeddings(),
            saliency_weight=saliency_weight,
            reinforcement_weight=reinforcement_weight,
            recency_weight=recency_weight,
            relevance_weight=relevance_weight,
            experiment_id=experiment_id,
        )

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        """
        Create an instance of the LLMChain class for generating text using the GPT-4 language model.

        Args:
            prompt (PromptTemplate): The prompt template to use for generating text.

        Returns:
            LLMChain: An instance of the LLMChain class.
        """
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    def create_memory_node(
        self,
        panas_scores: str,
        memory: str,
        memory_embedding,
        image_file_location: str,
        date: datetime,
        saliency_score: float,
        memory_number: int,
        contextual_understanding: str = "",
        situation: str = "",
    ) -> int:
        """
        Create a memory node for the new memory in the graph database.

        Args:
            panas_scores (str): The PANAS scores of the new memory.
            memory (str): The content of the new memory.
            memory_embedding (str): The memory embedding of the new memory.
            contextual_understanding (str): The contextual understanding of the new memory.
            image_file_location (str): The file location of the image.
            date (datetime): The date the memory was created.
            saliency_score (float): The saliency score of the memory.
            situation (Optional[str]): The situation of the memory (default None).
            memory_number (int): The number of the memory in the list of memories.

        Returns:
            int: The id of the new memory node.
        """
        with self.neo4j_driver.session() as session:
            node_id = session.run(
                """
                CREATE (node:Memory {
                    memory: $memory,
                    memory_embedding: $memory_embedding,
                    contextual_understanding: $contextual_understanding,
                    panas_scores: $panas_scores,
                    image_location: $image_file_location,
                    date: $date,
                    saliency_score: $saliency_score,
                    experiment_id: $experiment_id,
                    counter: 0,
                    situation: $situation,
                    memory_number: $memory_number
                })
                RETURN id(node) AS node_id
            """,
                memory=memory,
                memory_embedding=memory_embedding,
                contextual_understanding=contextual_understanding,
                panas_scores=panas_scores,
                image_file_location=image_file_location,
                saliency_score=saliency_score,
                experiment_id=self.experiment_id,
                date=date,
                situation=situation,
                memory_number=memory_number,
            ).single()["node_id"]
        return node_id

    def create_norm_node(self, norm_number: float, situation: str = "") -> int:
        """
        Create a norm node for the new norm in the graph database.

        Returns:
            int: The id of the new norm node.
        """

        with self.neo4j_driver.session() as session:
            node_id = session.run(
                """
                CREATE (node:Norm {
                    norm: $norm,
                    experiment_id: $experiment_id,
                    situation: $situation,
                    norm_number: $norm_number
                })
                RETURN id(node) AS node_id
            """,
                norm=self.norm,
                experiment_id=self.experiment_id,
                situation=situation,
                norm_number=norm_number,
            ).single()["node_id"]
        return node_id

    def create_norm_to_memory_edge(self, new_memory_id: int) -> None:
        """
        Create a relationship between the norm and the new memory in the Neo4j graph database.

        Args:
            new_memory_id (int): The ID of the new memory node to create a relationship with.

        Returns:
            None
        """

        query = """
            MATCH (newMemory:Memory), (norm:Norm)
            WHERE id(newMemory) = $new_memory_id AND id(norm)= $norm_id
            MERGE (norm)-[r:Influenced]->(newMemory)
        """
        parameters = {"new_memory_id": new_memory_id, "norm_id": self.norm_id}
        with self.neo4j_driver.session() as session:
            session.run(query, parameters=parameters)
        pass

    def create_memory_to_memory_edge(self, new_memory_id) -> None:
        """
        Create a relationship between the previous memory and the new memory in the Neo4j graph database.

        Args:
            new_memory_id (int): The ID of the new memory node to create a relationship with.

        Returns:
            None
        """

        query = """
            MATCH (previousMemory:Memory), (newMemoy:Memory)
            WHERE id(previousMemory) = $previous_memory_id AND id(newMemoy)= $new_memory_id
            MERGE (previousMemory)-[r:Next]->(newMemoy)
        """
        parameters = {"previous_memory_id": self.current_memory_id, "new_memory_id": new_memory_id}
        with self.neo4j_driver.session() as session:
            session.run(query, parameters=parameters)
        pass

        pass

    def create_memory_to_norm_edge(self) -> None:
        """
        Create a relationship between the memories that created the norm and the norm node in the Neo4j graph database.

        Returns:
            None
        """

        query = """
            UNWIND $norm_memories AS memory_id
            MATCH (memory:Memory), (norm:Norm)
            WHERE id(memory) = memory_id AND id(norm) = $norm_id
            MERGE (memory)-[r:Created]->(norm)
        """
        parameters = {
            "norm_memories": self.current_norm_ids,
            "norm_id": self.norm_id,
        }
        with self.neo4j_driver.session() as session:
            session.run(query, parameters=parameters)
        pass

    def find_all_memories(self) -> str:
        """
        Find all memories.

        Returns:
            str: The time and content of all memories, concatenated.
        """

        query = """
            MATCH (memory:Memory)
            WHERE memory.experiment_id = $experiment_id
            RETURN id(memory) AS node_id, memory.memory AS memory, memory.memory_number AS memory_number
            ORDER BY memory_number
        """
        with self.neo4j_driver.session() as session:
            results = session.run(query, parameters={"experiment_id": self.experiment_id}).data()

        result_strings = []
        norm_memory_ids = []

        for memory in results:
            formatted_string = f"{memory['memory']}"
            result_strings.append(formatted_string)
            norm_memory_ids.append(memory["node_id"])

        self.current_norm_ids = norm_memory_ids

        result_string = "\n".join(result_strings)

        return result_string

    def add_nodes_and_edges(
        self,
        panas_scores: str,
        memory: str,
        memory_number: int,
        date: datetime = datetime(2022, 1, 1),
        image_file_location: str = "",
        situation: str = "",
        memory_embedding: str = "",
        saliency_score: float = 0.0,
        contextual_understanding: str = "",
    ) -> List[str]:
        """
        Add memory and norm nodes to the graph database.

        Args:
            panas_scores (str): The PANAS scores of the memory to add.
            memory (str): The content of the memory to add.
            memory_embedding (str): The memory embedding of the memory to add.
            contextual_understanding (str): The contextual understanding of the memory to add.
            saliency_score (float): The saliency score of the memory to add.
            image_file_location (str): The file location of the image associated with the memory (default "").
            date (Optional[datetime]): The date of the memory (default None).
            situation (Optional[str]): The situation of the memory (default None).
            memory_number (int): The number of the memory in the list of memories.

        Returns:
            List[str]: A list containing the ID of the new memory node.
        """

        # Create Norm node
        self.norm_id = self.create_norm_node(situation=situation, norm_number=memory_number)

        # Create Memory node
        new_memory_id = self.create_memory_node(
            panas_scores=panas_scores,
            memory=memory,
            memory_embedding=memory_embedding,
            contextual_understanding=contextual_understanding,
            saliency_score=saliency_score,
            image_file_location=image_file_location,
            date=date,
            situation=situation,
            memory_number=memory_number,
        )

        # Connect the norm to the memory it influenced
        self.create_norm_to_memory_edge(new_memory_id)

        # Connect the memories which created the norm to the norm
        if self.current_norm_ids != []:
            self.create_memory_to_norm_edge()

        if self.current_memory_id is not None:
            # Connect the current memory to the memory it influenced
            self.create_memory_to_memory_edge(new_memory_id)

        self.current_memory_id = new_memory_id

        return new_memory_id

    def create_norm(self, memory_embedding) -> None:
        """
        Create the norm by finding similar memories, reinforcing their counters, finding the top memories with the largest counters,
        and then updating the norm using the GPT-4 language model.

        Args:
            memory_embedding (List[float]): The memory embedding of the new memory node.

        Returns:
            None
        """

        # 1. Similarity Search Using memory
        related_memory_ids = self.similarity_search(memory_embedding)

        # 2. Results counter += rho
        for memory_id in related_memory_ids:
            self.reinforce_memory(memory_id)

        # 3. Find memories with largest counter
        memories_in_norm: str = self.find_top_memories(memory_embedding)

        # 4. Update the norm using LLM
        prompt_summarise_memories = PromptTemplate.from_template(
            "The following are a list relevant memories and the corresponding time the memories were formed:\n"
            "---\n"
            "START MEMORIES\n"
            "{memories_in_norm}\n"
            "END MEMORIES\n"
            "---\n"
            "What high-level insights can you infer given the memories and the times they were formed?\n"
            "Are there time based trends?\n"
            "If there are no memories to extract insights from return 'No insights'.\n"
            "Example: \n"
            "---\n"
            "02/02/2014: A boy is watching a movie with his friends.\n"
            "03/02/2014: A boy is surrounded by his friends. He is happy.\n"
            "Insight: The boy hangs out with his friends once per month."
            "---\n"
        )

        self.norm = self.chain(prompt_summarise_memories).run(memories_in_norm=memories_in_norm)

    def create_norm_no_retrieval(self) -> None:
        """
        Create the norm by all memories.

        Returns:
            None
        """

        # Find all memories
        memories_in_norm: str = self.find_all_memories()

        # Update the norm using LLM
        prompt_summarise_memories = PromptTemplate.from_template(
            "The following are a list past memories:\n"
            "---\n"
            "START MEMORIES\n"
            "{memories_in_norm}\n"
            "END MEMORIES\n"
            "---\n"
            "What high-level insights can you infer given the memories?\n"
            "If there are no memories to extract insights from return 'No insights'.\n"
            "Example: \n"
            "---\n"
            "A boy is watching a movie with his friends.\n"
            "A boy is surrounded by his friends. He is happy.\n"
            "Insight: The boy often hangs out with his friends."
            "---\n"
        )

        self.norm = self.chain(prompt_summarise_memories).run(memories_in_norm=memories_in_norm)
