import copy
import re
import logging
from balrog.agents.base import BaseAgent
from balrog.client import LLMClientWrapper
from balrog.prompt_builder.history import Message
from balrog.environments.nle.base import NLELanguageWrapper
logger = logging.getLogger(__name__)

all_nle_action_map = NLELanguageWrapper.all_nle_action_map

available_actions = [
                action_strs[0]
                for _, action_strs in all_nle_action_map.items()
            ]
single_chars = [chr(i) for i in range(ord("a"), ord("z") + 1)] + [
                chr(i) for i in range(ord("A"), ord("Z") + 1)
            ]
single_digits = [str(i) for i in range(10)]
double_digits = [f"{i:02d}" for i in range(100)]
yes_no = ["yn", 'n']
all_actions = available_actions + single_chars + single_digits + double_digits + yes_no
all_actions_str = "\n-".join(all_actions)

class RobustCoTRAGAgent(BaseAgent):
    """An agent that performs actions using chain-of-thought reasoning with RAG-enabled retrieval."""

    def __init__(self, client_factory: LLMClientWrapper, prompt_builder, rag_instance, config):
        """Initialize the RobustCoTRAGAgent with a client, prompt builder, RAG instance, and configuration.

        Args:
            client_factory (LLMClientWrapper): A factory for creating the LLM client instance.
            prompt_builder (PromptBuilder): Object to build prompts for the agent.
            rag_instance: The RAG instance for retrieving relevant documents.
            config: Configuration object containing settings for the agent.
        """
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.rag = rag_instance
        self.remember_cot = config.agent.remember_cot
        logger.info("RobustCoTRAGAgent initialized")

    def act(self, obs, prev_action=None):
        """Generate the next action using chain-of-thought reasoning with RAG retrieval.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response containing the final selected action.
        """
        try:
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
                retrieved_docs = self.rag.search(query)
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

            # Combined instructions: RAG context + chain of thought + strict output format
            cot_rag_instructions = f"""
                                You are playing NetHack, a complex dungeon-crawling game. Use the retrieved context to inform your decision.

                                1. **Analyze the Situation**: Examine the current game state, including your inventory, position, and any visible threats or opportunities.

                                2. **Use Retrieved Context**: The retrieved documents provide insights into the game's environment and potential actions.

                                3. **Decide on an Action**: Choose the best course of action based on the analysis and context.

                                4. **Yes/No**: If the action is a yes/no question, you must output yn or n.

                                5. **Output the Action**: You must output the action strictly in the format:

                                <|ACTION|>YOUR_CHOSEN_ACTION<|END|>

                                Replace YOUR_CHOSEN_ACTION with one of the following valid actions:
                                - {all_actions_str}

                                Ensure the action is valid within the context of NetHack. Do not include any additional text or reasoning in your response.
                                """.strip()

            if messages and messages[-1].role == "user":
                messages[-1].content += "\n\n" + cot_rag_instructions
                logger.debug("Added CoT-RAG instructions to final message")

            # Log the final prompt content
            logger.debug("Sending prompt to LLM client")
            for msg in messages:
                logger.debug(f"Message {msg.role}: {msg.content[:100]}...")

            # Generate the CoT reasoning
            cot_reasoning = self.client.generate(messages)
            logger.debug(f"Received response from LLM: {cot_reasoning}")

            # Extract the final answer from the CoT reasoning
            final_answer = self._extract_final_answer(cot_reasoning)
            logger.debug(f"Extracted final answer: {final_answer}")

            return final_answer

        except Exception as e:
            logger.error(f"Error in act(): {str(e)}", exc_info=True)
            # Return a safe default response in case of error
            return self.client.generate([Message(role="user", content="Output a single valid action in the format <|ACTION|>action<|END|>.")])

    def _extract_final_answer(self, reasoning):
        """Extract the final action from the chain-of-thought reasoning response.

        Args:
            reasoning (LLMResponse): The response containing CoT reasoning and action.

        Returns:
            LLMResponse: The response with the extracted final action in `completion`
                         and the entire chain-of-thought in `reasoning`.
        """
        try:
            final_answer = copy.deepcopy(reasoning)

            final_answer = final_answer._replace(reasoning=reasoning.completion)

            completion_text = reasoning.completion
            match = re.search(r"<\|ACTION\|>(.*?)<\|END\|>", completion_text, re.DOTALL)
            if match:
                extracted_action = match.group(1).strip()
            else:
                logger.warning("Failed to extract action using the strict format")
                extracted_action = self._fallback_extraction(completion_text)

            # Replace the final `completion` with only the extracted action
            final_answer = final_answer._replace(completion=extracted_action)

            return final_answer
            
        except Exception as e:
            logger.error(f"Error in _extract_final_answer(): {str(e)}")
            return reasoning

    def _fallback_extraction(self, text):
        """Fallback method to extract an action when the strict format fails."""
        # Filter to keep only alphabetic characters as a last resort
        return re.sub(r"[^a-zA-Z\s:]", "", text).strip()