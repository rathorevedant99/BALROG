from balrog.client import create_llm_client

from ..prompt_builder import create_prompt_builder
from .chain_of_thought import ChainOfThoughtAgent
from .custom import CustomAgent
from .dummy import DummyAgent
from .few_shot import FewShotAgent
from .naive import NaiveAgent
from .robust_naive import RobustNaiveAgent
from .robust_cot import RobustCoTAgent
from .naive_rag import NaiveRAGAgent
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
        self.rag_instance = None
        if hasattr(self.config, 'rag') and hasattr(self.config.rag, 'enabled') and self.config.rag.enabled:
            self._initialize_rag()

    def _initialize_rag(self):
        """Initialize the RAG instance if RAG is enabled in config."""
        if not hasattr(self.config.rag, 'model_name') or not hasattr(self.config.rag, 'documents_path'):
            raise ValueError("RAG configuration must include 'model_name' and 'documents_path'")
        
        # Get device from config, default to 'cuda'
        device = getattr(self.config.rag, 'device', 'cuda')
        logger.info(f"Initializing RAG with device: {device}")
        
        # Initialize RAG with explicit device parameter
        self.rag_instance = RAG(
            model_name=self.config.rag.model_name,
            device=device,  # Pass device explicitly
            cache_dir=getattr(self.config.rag, 'cache_dir', './rag_cache')
        )
        documents = self.load_documents(self.config.rag.documents_path)
        self.rag_instance.build_index(documents)

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
        elif path.endswith('.txt'):
            return parse_txt(path)
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
            "naive_rag": lambda: NaiveRAGAgent(client_factory, prompt_builder, self.rag_instance)
        }

        agent_type = self.config.agent.type
        if agent_type not in agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")

        if agent_type == "naive_rag" and self.rag_instance is None:
            raise ValueError("RAG must be enabled in config to use naive_rag agent")

        return agent_types[agent_type]()
