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

            context = obs["text"]["short_term_context"]
            # query = obs["text"].get("long_term_context", "")

            # dungeon_level_match = re.search(r"Dlvl:(\d+)", context)
            # xp_level_match = re.search(r"Xp:(\d+)", context)
            
            # current_dungeon_level = dungeon_level_match.group(1) if dungeon_level_match else "unknown"
            # current_xp_level = xp_level_match.group(1) if xp_level_match else "unknown"
            
            # # Extract HP information
            # hp_match = re.search(r"HP:(\d+)\((\d+)\)", context)
            # current_hp = hp_match.group(1) if hp_match else None
            # max_hp = hp_match.group(2) if hp_match else None
            
            # Reset query count if context changed
            if self.last_context != context:
                self.query_count = 0
                self.last_context = context
            else:
                self.query_count += 1

            # Extract key information from context
            inventory_items = re.findall(r"[a-zA-Z]\s-\s([^\n]+)", context)
            visible_items = re.findall(r"You see here ([^\n]+)", context)
            monsters = re.findall(r"You see ([^.]+)\.", context)
            
            # Define different query templates based on query count
            query_templates = [
                # Combat and monsters
                f"""QUESTION: {' '.join(monsters)} combat strategy attributes weaknesses""",
                
                # Item identification and usage
                f"""QUESTION: {' '.join(inventory_items + visible_items)} uses effects benefits""",
                
                # Equipment and inventory optimization
                f"""QUESTION: optimal equipment loadout {' '.join(inventory_items)}""",
                
                # Dungeon features and navigation
                """QUESTION: dungeon features corridors doors traps navigation""",
                
                # Survival and status management
                """QUESTION: HP management healing recovery survival tactics"""

                # Exploration and discovery
                """QUESTION: explore new areas discover hidden paths progress"""
            ]
            
            current_template = query_templates[self.query_count % len(query_templates)]
            
            interim_query = f"""
            Based on the current game state: {context}

            Generate a SHORT, FOCUSED search query (2-4 keywords) related to:
            {current_template}

            Focus on SPECIFIC ITEMS, MONSTERS, or FEATURES currently visible.
            DO NOT ask questions - use keywords only.
            
            Previous queries focused on: {', '.join(query_templates[:(self.query_count % len(query_templates))])}
            
            Reply in the form of: QUESTION: <keywords>
            """.strip()

            # """{context}\n\n
            # Asses the current situation properly. There is an available RAG document store that has all the information about the game NetHack.
            # Given the situation, ask a short question that you think will help you learn more about the game, inventory items, monsters or anything
            # that will help you make a decision. Your question should not be more than 4-5 words. Your question should be a question that you think will help you make a decision.
            # Your question should not ask about the general game mechanics."""


            # interim_query = f"""You are currently on dungeon level {current_dungeon_level} and have {current_xp_level} experience points. Your goal is to 
            # maximize your dungeon level and experience points. To do so, you must explore the dungeons, fight monsters, and collect items. Examine the current game state, 
            # including your inventory, position, and any visible threats or opportunities, you can ask a question to a large document store to get more information about the dungeon and the actions you should perform. Knowing all this,
            # create a short query to retrive the most relevant information from the document store. Your query should leverage your current observations and inventory to 
            # get the most relevant information that you can use immediately and help you plan for the future. The query should be pinpointed and not
            # general in nature.
            # """


            query = self.client.generate([Message(role="user", content=interim_query)])
            query = query.completion
            query = re.search(r"QUESTION: (.*)", query)
            query = query.group(1) if query else ""
            logger.info(f"Generated RAG query: {query[:100]}...")  # Log first 100 chars of query

            try:
                # Retrieve relevant documents using RAG
                retrieved_docs = self.rag.search(query)
                logger.info(f"Retrieved {len(retrieved_docs)} documents")
                
                processed_docs = [doc for doc, _ in retrieved_docs]
                self.prompt_builder.update_retrieved_docs(processed_docs)
                logger.info(f"Processed {len(processed_docs)} relevant documents")
                # refine_prompt = f"""
                # For the prompt: {query}
                # The retrieved documents are:
                # {processed_docs}
                # Summarize the retrieved documents in a concise manner so that it can be used by a game 
                # player to make decisions. Reply in the form of: SUMMARY: <summary>
                # """
                # processed_rag = self.client.generate([Message(role="user", content=refine_prompt)])
                # processed_rag = processed_rag.completion
                # logger.debug(f"Raw processed_rag output: {repr(processed_rag)}")
                
                # # Extract summary if it exists
                # summary_match = re.search(r"SUMMARY:\s*(.*)", processed_rag)
                # processed_rag = summary_match.group(1).strip() if summary_match else processed_rag.strip()
                # processed_rag = [processed_rag]
                # self.prompt_builder.update_retrieved_docs(processed_rag)

            except Exception as e:
                logger.error(f"Error during RAG retrieval: {str(e)}")
                # Continue without retrieved docs if RAG fails
                processed_docs = []

            messages = self.prompt_builder.get_prompt()
            logger.debug(f"Generated {len(messages)} messages for prompt")

            # Combined instructions: RAG context + chain of thought + strict output format
            cot_rag_instructions = f"""
                                Use the retrieved context to inform your decision. It's mentioned in the content in the "Relevant Context from RAG:" section. The goal is to
                                explore the dungeon, kill monsters and survive. This can be measured by the dungeon level and the experience points.

                                1. The retrieved context might not be totally accurate. Use your own understanding along with the retrieved context to make a decision.

                                2. **Plan for the future**: The final goal is achieved by intermediary steps. Plan to achieve the final goal by taking a series of steps. But at one time, you can only take one action. So only show the next action in the plan.

                                3. **Decide on an Action**: Choose the best course of action based on the analysis and context and final goal of the plan.

                                4. **Yes/No**: If the action is a yes/no question, you must output yn or n.

                                5. **Output the Action**: You must output the action strictly in the format:

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
                                - Unexplored areas on the map are dark and will not have any ascii characters. These areas can be explored by moving in that direction. If you are in such an area,
                                and can't move in one direction, try to move in another direction. When you explore a path, you will see ascii # characters on the map.
                                - Walls are marked with an underscore "_" horizontally and a pipe "|" vertically. If through a set of walls you see a space, it means you can move through that space.
                                It might be a door or a passage.
                                - If the observations show a certain item or object at a particular location, you must first move to that location to interact with it. For example, 
                                if there is a door far west, you must first move west and then open the door.
                                - Eating while satiated will lead to choking and death. Do not eat when satiated.
                                """.strip()
            # - Any stairs will lead to a new level. Do not confine yourself to only going up or down. Use the stairs to explore other levels regardless of the direction.

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