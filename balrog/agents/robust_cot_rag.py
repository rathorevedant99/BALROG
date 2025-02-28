import copy
import re
import logging
from balrog.agents.base import BaseAgent
from balrog.client import LLMClientWrapper
from balrog.prompt_builder.history import Message
from balrog.environments.nle.base import NLELanguageWrapper
from nle.nethack import USEFUL_ACTIONS
logger = logging.getLogger(__name__)

all_nle_action_map = NLELanguageWrapper.all_nle_action_map

available_actions = [
                action_strs[0]
                for action, action_strs in all_nle_action_map.items()
                if action in USEFUL_ACTIONS
            ]
single_chars = [chr(i) for i in range(ord("a"), ord("z") + 1)] + [
                chr(i) for i in range(ord("A"), ord("Z") + 1)
            ]
single_digits = [str(i) for i in range(10)]
double_digits = [f"{i:02d}" for i in range(100)]
yes_no = ["yn", 'n']
all_actions = available_actions + single_chars + single_digits + double_digits + yes_no
all_actions_str = "\n-".join(all_actions)

ACTIONS = {
    "north": "move north",
    "east": "move east",
    "south": "move south",
    "west": "move west",
    "northeast": "move northeast",
    "southeast": "move southeast",
    "southwest": "move southwest",
    "northwest": "move northwest",
    "far north": "move far north",
    "far east": "move far east",
    "far south": "move far south",
    "far west": "move far west",
    "far northeast": "move far northeast",
    "far southeast": "move far southeast",
    "far southwest": "move far southwest",
    "far northwest": "move far northwest",
    "up": "go up a staircase",
    "down": "go down a staircase (tip: you can only go down if you are standing on the stairs)",
    "wait": "rest one move while doing nothing",
    "more": "display more of the message (tip: ONLY ever use when current message ends with --More--)",
    "annotate": "leave a note about the level",
    "apply": "apply (use) a tool",
    "call": "name a monster or object, or add an annotation",
    "cast": "cast a spell",
    "close": "close an adjacent door",
    "open": "open an adjacent door",
    "dip": "dip an object into something",
    "drop": "drop an item",
    "droptype": "drop specific item types (specify in the next prompt)",
    "eat": "eat something (tip: replenish food when hungry)",
    "esc": "exit menu or message",
    "engrave": "engrave writing on the floor (tip: Elbereth)",
    "enhance": "advance or check weapons skills",
    "fire": "fire ammunition from quiver",
    "fight": "fight a monster (even if you only guess one is there)",
    "force": "force a lock",
    "inventory": "show your inventory",
    "invoke": "invoke ",
    "jump": "jump to a location",
    "kick": "kick an enemy or a locked door or chest",
    "look": "look at what is under you",
    "loot": "loot a box on the floor",
    "monster": "use a monster's special ability (when polymorphed)",
    "offer": "offer a sacrifice to the gods (tip: on an aligned altar)",
    "overview": "display an overview of the dungeon",
    "pay": "pay your shopping bill",
    "pickup": "pick up things at the current location",
    "pray": "pray to the gods for help",
    "puton": "put on an accessory",
    "quaff": "quaff (drink) something",
    "quiver": "select ammunition for quiver",
    "read": "read a scroll or spellbook",
    "remove": "remove an accessory",
    "rub": "rub a lamp or a stone",
    "search": "search for hidden doors and passages",
    "swap": "swap wielded and secondary weapons",
    "takeoff": "take off one piece of armor",
    "takeoffall": "take off all armor",
    "teleport": "teleport to another level (if you have the ability)",
    "throw": "throw something (e.g. a dagger or dart)",
    "travel": "travel to a specific location on the map (tip: in the next action, specify > or < for stairs, { for fountain, and _ for altar)",
    "twoweapon": "toggle two-weapon combat",
    "untrap": "untrap something",
    "wear": "wear a piece of armor",
    "wield": "wield a weapon",
    "wipe": "wipe off your face",
    "zap": "zap a wand",
    "minus": "-",
    "space": " ",
    "apos": "'",
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
}

action_strings = ",\n".join(f"{action}: {description}" for action, description in ACTIONS.items())

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
            # short_term = obs["text"]["short_term_context"]
            # long_term = obs["text"].get("long_term_context", "")
            # query = f"{short_term} {long_term}".strip()

            query = obs["text"]["short_term_context"]
            # query = obs["text"].get("long_term_context", "")
            
            logger.info(f"Generated query: {query[:100]}...")  # Log first 100 chars of query

            try:
                # Retrieve relevant documents using RAG
                retrieved_docs = self.rag.search(query)
                logger.info(f"Retrieved {len(retrieved_docs)} documents")
                
                processed_docs = [doc for doc, _ in retrieved_docs]
                logger.info(f"Processed {len(processed_docs)} relevant documents")
                # self.prompt_builder.update_retrieved_docs(processed_docs)

                ## Show only the first 500 characters of the content
                # processed_text = "\n".join(
                #     [f"Title: {doc[0]}\nContent: {doc[1][:500]}" for doc in processed_docs]
                # )
                # self.prompt_builder.update_retrieved_docs(processed_text)

                self.prompt_builder.update_retrieved_docs(processed_docs)

            except Exception as e:
                logger.error(f"Error during RAG retrieval: {str(e)}")
                # Continue without retrieved docs if RAG fails
                processed_docs = []

            messages = self.prompt_builder.get_prompt()
            logger.debug(f"Generated {len(messages)} messages for prompt")

            # Combined instructions: RAG context + chain of thought + strict output format
            cot_rag_instructions = f"""
                                Use the retrieved context to inform your decision. It's mentioned in the content in the "Relevant Context from RAG:" section.

                                1. **Analyze the Situation**: Examine the current game state, including your inventory, position, and any visible threats or opportunities.

                                2. **Use the retrieved context to inform your decision**: The retrieved documents provide insights into the game's environment and potential actions.

                                3. **Plan for the future**: The final goal is achieved by intermediary steps. Plan to achieve the final goal by taking a series of steps. But at one time, you can only take one action. So only show the next action in the plan.

                                4. **Decide on an Action**: Choose the best course of action based on the analysis and context and final goal of the plan.

                                5. **Yes/No**: If the action is a yes/no question, you must output yn or n.

                                6. **Output the Action**: You must output the action strictly in the format:

                                <|ACTION|>YOUR_CHOSEN_ACTION<|END|>

                                Replace YOUR_CHOSEN_ACTION with one of the valid actions provided in the list of actions mention at the beginning of the prompt.
                                Use the action that is mentioned before the colon in the list. Do not use the action description mentioned after the colon.

                                Ensure the action is valid within the context of NetHack. Your response should start with <|ACTION|>YOUR_CHOSEN_ACTION<|END|>.
                                The chosen action should be the one that is a strong move to achieve the final goal. The chosen action can only be from the list of actions otherwise you will not be able to perform the action.
                                You can only output one action at a time. Do not create a combination of actions. Only one action from the list of actions should be output.
                                After that, you can include a short reasoning in your response. For example, if you want to open a door in the direction of north, you can output <|ACTION|>open<|END|>. And then in the 
                                next call, you can output <|ACTION|>north<|END|>.

                                Important tips:
                                - When executing an action, the message will give you the result of the action. Make sure to read the message carefully to understand the result of the action and if the action was successful.
                                - Any stairs will lead to a new level. Do not confine yourself to only going up or down. Use the stairs to explore other levels regardless of the direction.
                                - Unexplored areas on the map are dark and will not have any ascii characters. These areas can be explored by moving in that direction. If you are in such an area,
                                and can't move in one direction, try to move in another direction. When you explore a path, you will see ascii # characters on the map.
                                - Walls are marked with an underscore "_" horizontally and a pipe "|" vertically. If through a set of walls you see a space, it means you can move through that space.
                                It might be a door or a passage.
                                - If the observations show a certain item or object at a particular location, you must first move to that location to interact with it. For example, 
                                if there is a door far west, you must first move west and then open the door.
                                - Eating while satiated will lead to choking and death. Do not eat when satiated.
                                """.strip()

            if messages and messages[-1].role == "user":
                messages[-1].content += "\n\n" + cot_rag_instructions
                logger.debug("Added CoT-RAG instructions to final message")

            # Log the final prompt content
            logger.debug("Sending prompt to LLM client")
            for msg in messages:
                logger.info(f"Message {msg.role}: {msg.content}...")

            # Generate the CoT reasoning
            cot_reasoning = self.client.generate(messages)
            logger.info(f"Received response from LLM: {cot_reasoning}")

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