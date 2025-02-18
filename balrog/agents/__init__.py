from balrog.client import create_llm_client

from ..prompt_builder import create_prompt_builder
from .chain_of_thought import ChainOfThoughtAgent
from .custom import CustomAgent
from .dummy import DummyAgent
from .few_shot import FewShotAgent
from .naive import NaiveAgent
from .robust_naive import RobustNaiveAgent
from .robust_cot import RobustCoTAgent
from .naive_rag_vedant import NaiveRAGAgent
from .naive_rag import RAGNaiveAgent
from .utils.rag import RAG, parse_xml

import logging
logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory class for creating agents based on configuration.

    The `AgentFactory` class is responsible for initializing the appropriate agent type
    based on the provided configuration, which includes setting up the LLM client and
    prompt builder.
    """

    def __init__(self, config):
        """Initialize the AgentFactory with configuration settings.

        Args:
            config (omegaconf.DictConfig): Configuration object containing settings for the agent and client.
        """
        self.config = config
        self.rag_config = None
        if hasattr(self.config, 'rag') and hasattr(self.config.rag, 'enabled') and self.config.rag.enabled:
            self.rag_config = self.config
            # Pre-load documents to avoid multiple loads
            self.documents = self.load_documents(self.config.rag.documents_path)

    def _create_rag_instance(self):
        """Create a new RAG instance when needed."""
        if not self.rag_config:
            raise ValueError("RAG configuration not initialized")
            
        logger.info(f"Creating new RAG instance with device: {self.rag_config.rag.device}")
        rag_instance = RAG(self.rag_config)
        # Build index using pre-loaded documents
        rag_instance.build_index(self.documents)
        return rag_instance

    def load_documents(self, path):
        """Load and parse documents from XML or TXT files.

        Args:
            path (str): Path to the document file.

        Returns:
            list: List of document contents.

        Raises:
            ValueError: If the file extension is not supported.
        """
        if path.endswith('.xml'):
            return parse_xml(path)
        else:
            raise ValueError(f"Unsupported document format: {path}")

    def create_agent(self):
        """Create an agent instance based on the configuration.

        Returns:
            Agent: An instance of the selected agent type.

        Raises:
            ValueError: If an unknown agent type is specified.
        """
        client_factory = create_llm_client(self.config.client)
        prompt_builder = create_prompt_builder(self.config.agent)

        agent_types = {
            "naive": lambda: NaiveAgent(client_factory, prompt_builder),
            "cot": lambda: ChainOfThoughtAgent(client_factory, prompt_builder, config=self.config),
            "dummy": lambda: DummyAgent(client_factory, prompt_builder),
            "custom": lambda: CustomAgent(client_factory, prompt_builder),
            "few_shot": lambda: FewShotAgent(client_factory, prompt_builder, self.config.agent.max_icl_history),
            "robust_naive": lambda: RobustNaiveAgent(client_factory, prompt_builder),
            "robust_cot": lambda: RobustCoTAgent(client_factory, prompt_builder, config=self.config),
            # "naive_rag": lambda: RAGNaiveAgent(client_factory, prompt_builder, self.rag_instance)
            "naive_rag": lambda: RAGNaiveAgent(client_factory, prompt_builder, config=self.config),
            "naive_rag_vedant": lambda: NaiveRAGAgent(
                client_factory, 
                prompt_builder, 
                self._create_rag_instance()  # Create new RAG instance for each agent
            )
        }

        agent_type = self.config.agent.type
        if agent_type not in agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")

        if agent_type == "naive_rag_vedant" and self.rag_config is None:
            raise ValueError("RAG must be enabled in config to use naive_rag agent")

        return agent_types[agent_type]()
