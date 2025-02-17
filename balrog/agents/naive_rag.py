import copy
import re
from balrog.agents.base import BaseAgent

class NaiveRAGAgent(BaseAgent):
    """An agent that generates actions based on observations with RAG-enabled retrieval."""

    def __init__(self, client_factory, prompt_builder, rag_instance):
        """Initialize the NaiveRAGAgent with a client, prompt builder, and RAG instance."""
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.rag = rag_instance

    def act(self, obs, prev_action=None):
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            str: The selected action from the LLM response.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        # Get the query from the observation
        query = obs["text"]["short_term_context"]

        # Retrieve relevant documents using RAG
        retrieved_docs = self.rag.search(query)
        self.prompt_builder.update_retrieved_docs([doc for doc, _ in retrieved_docs])

        messages = self.prompt_builder.get_prompt()

        naive_instruction = """
You always have to output one of the above actions at a time and no other text. You always have to output an action until the episode terminates.
        """.strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction

        response = self.client.generate(messages)

        final_answer = self._extract_final_answer(response)

        return final_answer

    def _extract_final_answer(self, answer):
        """Sanitize the final answer, keeping only alphabetic characters.

        Args:
            answer (LLMResponse): The response from the LLM.

        Returns:
            LLMResponse: The sanitized response.
        """
        def filter_letters(input_string):
            return re.sub(r"[^a-zA-Z\s:]", "", input_string)

        final_answer = copy.deepcopy(answer)
        final_answer = final_answer._replace(completion=filter_letters(final_answer.completion))

        return final_answer