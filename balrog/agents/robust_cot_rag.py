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
        self.last_context = None
        self.query_count = 0  # Track how many times we've queried the same context
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

            short_term_context = obs["text"]["short_term_context"]
            long_term_context = obs["text"].get("long_term_context", "")
            context = f"{short_term_context} {long_term_context}".strip()
            logger.debug(f"Context: {context}")

            system_prompt = self.prompt_builder.system_prompt

            system_prompt += context


            # """Your primary goal is to explore the dungeon,
            # kill monsters and survive. This can be measured by the dungeon level and the experience points."""
            # rag_query_prompt = system_prompt + \
            # """
            # Understand the provided context for your current state and map position. 

            # You can retrieve relevant information from a RAG database. Given the current state and map position, you can ask a keyword
            # query to retrieve relevant information. Example: "Magic Potion Usage Effects"

            # The RAG database is to help you find documents that explain different 

            # Create a short rag query of not more than 4 words given your current state. Respond in the format:
            # Query:<query>
            # """

            rag_query_prompt = system_prompt + \
            """
            Look at the inventory and map properly. Now imagine that you have a information rich document for the game NetHack that has information about game mechanics and optimal strategies. The document also has information about the characters in game and the their abilities. It also has information about the weapons or objects that you find in the game.
            Output a concise 4-5 words sentence of what you would like to get from the document. For example: "fountain", or "defeat a fox?
            Respond in the format:
            Query:<query>
            """

            logger.debug(f"RAG Query Prompt:{rag_query_prompt}")

            rag_response = self.client.generate([Message(role="user", content=rag_query_prompt)])
            rag_query = rag_response.completion
            rag_query = rag_query.split("Query:")[1].strip()

            logger.info(f"RAG query: {rag_query}")

            rag_docs = self.rag.search(rag_query)
            
            rag_context = "\n".join([doc for doc, _ in rag_docs])

            logger.debug(f"RAG context: {rag_context}")

            rag_usage_prompt = system_prompt + context + \
            f"""
            Below is the retrieved context from the RAG database. Use this information to help you make a decision.
            {rag_context}
            """

            self.prompt_builder.update_instruction_prompt(rag_usage_prompt)

            cot_instructions = """
                                Given the retrieved context, think step-by-step to what will help progress towards the goal.
                                Then, you must choose exactly one of the listed actions and output it strictly in the following format:

                                <|ACTION|>YOUR_CHOSEN_ACTION<|END|>

                                Replace YOUR_CHOSEN_ACTION with the chosen action.
                                
                                The chosen action can only be from the list of actions provided. 
                                
                                Additional tips:
                                - Yes or no can be responded with yn or n
                                - Anything which is not from the list of actions, is not a valid action
                                - You should devise a strategy basis your current state and the retrieved context. For example, items that are to be used at different levels, or different monsters that you can defeat
                                - Planning for future is very helpful. For example, if you need to defeat a monster, you can plan for that by saving items or weapons that you can use later""".strip()
            
            messages = self.prompt_builder.get_prompt()
            messages[-1].content += "\n\n" + cot_instructions

            logger.info(f"COT Prompt: {messages}")

            cot_reasoning = self.client.generate(messages)

            logger.info(f"COT reasoning: {cot_reasoning}")

            final_answer = self._extract_final_answer(cot_reasoning)

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