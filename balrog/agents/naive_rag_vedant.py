import copy
import re
import logging
from balrog.agents.base import BaseAgent

logger = logging.getLogger(__name__)

class NaiveRAGAgent(BaseAgent):
    """An agent that generates actions based on observations with RAG-enabled retrieval."""

    def __init__(self, client_factory, prompt_builder, rag_instance):
        """Initialize the NaiveRAGAgent with a client, prompt builder, and RAG instance."""
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.rag = None
        self._rag_instance = rag_instance
        logger.info("NaiveRAGAgent initialized")

    def act(self, obs, prev_action=None):
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            str: The selected action from the LLM response.
        """
        try:
            if self.rag is None:
                self.rag = self._rag_instance
                logger.info("RAG instance initialized in act()")
            
            if prev_action:
                self.prompt_builder.update_action(prev_action)
                logger.debug(f"Previous action updated: {prev_action}")

            self.prompt_builder.update_observation(obs)
            logger.debug("Observation updated")

            # Get the query from the observation - combine both contexts for better retrieval
            short_term = obs["text"]["short_term_context"]
            long_term = obs["text"].get("long_term_context", "")
            query = f"{short_term} {long_term}".strip()
            
            logger.debug(f"Generated query: {query[:100]}...")  # Log first 100 chars of query

            try:
                # Retrieve relevant documents using RAG
                retrieved_docs = self.rag.search(query, top_k=10)
                logger.info(f"Retrieved {len(retrieved_docs)} documents")
                
                # Filter and process retrieved documents
                processed_docs = []
                for doc, score in retrieved_docs:
                    if score < 1.5:  # Only include relevant documents
                        doc = doc.strip()
                        if doc:
                            processed_docs.append(doc)
                            logger.debug(f"Added doc with score {score}: {doc[:100]}...")

                logger.info(f"Processed {len(processed_docs)} relevant documents")
                self.prompt_builder.update_retrieved_docs(processed_docs)

            except Exception as e:
                logger.error(f"Error during RAG retrieval: {str(e)}")
                # Continue without retrieved docs if RAG fails
                processed_docs = []

            messages = self.prompt_builder.get_prompt()
            logger.debug(f"Generated {len(messages)} messages for prompt")

            # Enhanced instruction that emphasizes using the retrieved context
            rag_instruction = """
Consider the retrieved context carefully when deciding your action. The context contains relevant information about game mechanics and optimal strategies.
You always have to output one of the above actions at a time and no other text. You always have to output an action until the episode terminates.
            """.strip()

            if messages and messages[-1].role == "user":
                messages[-1].content += "\n\n" + rag_instruction
                logger.debug("Added RAG instruction to final message")

            # Log the final prompt content
            logger.debug("Sending prompt to VLLM client")
            for msg in messages:
                logger.debug(f"Message {msg.role}: {msg.content[:100]}...")

            response = self.client.generate(messages)
            logger.debug(f"Received response from VLLM: {response}")

            final_answer = self._extract_final_answer(response)
            logger.debug(f"Extracted final answer: {final_answer}")

            return final_answer

        except Exception as e:
            logger.error(f"Error in act(): {str(e)}", exc_info=True)
            # Return a safe default response in case of error
            return self.client.generate([Message(role="user", content="Output a single valid action.")])

    def _extract_final_answer(self, answer):
        """Sanitize the final answer, keeping only alphabetic characters."""
        try:
            def filter_letters(input_string):
                return re.sub(r"[^a-zA-Z\s:]", "", input_string)

            final_answer = copy.deepcopy(answer)
            final_answer = final_answer._replace(completion=filter_letters(final_answer.completion))
            return final_answer

        except Exception as e:
            logger.error(f"Error in _extract_final_answer(): {str(e)}")
            return answer