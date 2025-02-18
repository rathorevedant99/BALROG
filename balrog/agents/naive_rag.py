import copy
import re

from balrog.agents.base import BaseAgent
from balrog.agents.agent_rag_utils import *


class RAGNaiveAgent(BaseAgent):
    """An agent that generates actions guided by RAG based on observations without complex reasoning."""

    def __init__(self, client_factory, prompt_builder, config):
        """Initialize the RAGNaiveAgent with a client, prompt builder, and retriever."""
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.retriever = NethackWikiSearch(config)
        self.retriever.load_index()

    def act(self, obs, prev_action=None):
        """Generate the next action based on the observation and retrieved context.

        Args:
            obs (dict): The current observation.
            prev_action (str, optional): The previous action taken.

        Returns:
            str: The selected action from the LLM response.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        query = obs["text"]["short_term_context"]  # Use short-term context as the query. In practise this is not good idea
        retrieved_docs = self.retriever.search(query)
        print(query) # Look at these later in eval
        print("\n".join([f"{title}: {content[:1]}" for title, content in retrieved_docs])) # Retrieved titles, look at these later in eval

        self.prompt_builder.update_observation(obs) # Retrieved docs are not part of the observation (yet?)

        messages = self.prompt_builder.get_prompt()

        naive_instruction = """
You always have to output one of the above actions at a time and no other text. You always have to output an action until the episode terminates.
        """.strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction

        # Format retrieved documents nicely
        retrieved_text = "\n".join(
            [f"Title: {doc[0]}\nContent: {doc[1][:500]}\n{'-'*40}" for doc in retrieved_docs] # Give first 500 chars of each doc for now
        )

        rag_instruction = f"""
    Use these documents to help you decide your next action:
    {retrieved_text}
        """

        # Append retrieved docs to the last user message
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + rag_instruction

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