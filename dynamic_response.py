import logging
import random
from typing import Dict, Any, Optional
import config

# Configure logging
logger = logging.getLogger(__name__)

class DynamicResponseManager:
    """
    Class to handle dynamic response length and style
    """
    def __init__(self):
        """Initialize the dynamic response manager"""
        self.last_response_type = None
        self.consecutive_same_type_count = 0
        logger.info("Dynamic response manager initialized")

    def get_response_type(self, message_content: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Determine the type of response to generate based on probabilities and context

        Args:
            message_content: The user's message content
            context: Optional context information about the conversation

        Returns:
            Response type: "extremely_short", "slightly_short", "medium", "slightly_long", or "long"
        """
        if not config.DYNAMIC_MESSAGE_LENGTH_ENABLED:
            return "medium"  # Default to medium if dynamic length is disabled

        # Base probabilities from config
        probabilities = {
            "extremely_short": config.EXTREMELY_SHORT_RESPONSE_PROBABILITY,
            "slightly_short": config.SLIGHTLY_SHORT_RESPONSE_PROBABILITY,
            "medium": config.MEDIUM_RESPONSE_PROBABILITY,
            "slightly_long": config.SLIGHTLY_LONG_RESPONSE_PROBABILITY,
            "long": config.LONG_RESPONSE_PROBABILITY
        }

        # Adjust probabilities based on message content
        self._adjust_probabilities_for_content(probabilities, message_content)

        # Adjust probabilities based on conversation context
        if context:
            self._adjust_probabilities_for_context(probabilities, context)

        # Adjust probabilities to avoid repetitive patterns
        self._adjust_probabilities_for_variety(probabilities)

        # Apply randomness factor
        self._apply_randomness(probabilities)

        # Normalize probabilities
        total = sum(probabilities.values())
        normalized_probabilities = {k: v/total for k, v in probabilities.items()}

        # Select response type based on probabilities
        response_type = self._select_response_type(normalized_probabilities)

        # Update tracking variables
        if response_type == self.last_response_type:
            self.consecutive_same_type_count += 1
        else:
            self.consecutive_same_type_count = 0
            self.last_response_type = response_type

        logger.info(f"Selected response type: {response_type}")
        return response_type

    def _adjust_probabilities_for_content(self, probabilities: Dict[str, float], message_content: str) -> None:
        """
        Adjust probabilities based on the user's message content

        Args:
            probabilities: The current probability distribution
            message_content: The user's message content
        """
        # Short messages tend to get shorter responses
        if len(message_content) < 50:
            probabilities["extremely_short"] *= 1.5
            probabilities["slightly_short"] *= 1.3
            probabilities["long"] *= 0.7

        # Long, complex messages tend to get longer responses
        elif len(message_content) > 200:
            probabilities["slightly_long"] *= 1.3
            probabilities["long"] *= 1.5
            probabilities["extremely_short"] *= 0.7

        # Questions often get medium-length responses
        if "?" in message_content:
            probabilities["medium"] *= 1.3
            probabilities["slightly_long"] *= 1.2

        # Commands or requests often get short confirmations
        command_indicators = ["please", "can you", "could you", "would you", "tell me", "show me", "help me"]
        if any(indicator in message_content.lower() for indicator in command_indicators):
            probabilities["extremely_short"] *= 1.2
            probabilities["slightly_short"] *= 1.1

    def _adjust_probabilities_for_context(self, probabilities: Dict[str, float], context: Dict[str, Any]) -> None:
        """
        Adjust probabilities based on conversation context

        Args:
            probabilities: The current probability distribution
            context: Context information about the conversation
        """
        # If this is the first message in a conversation, tend toward medium or slightly long
        if context.get("is_first_message", False):
            probabilities["medium"] *= 1.5
            probabilities["slightly_long"] *= 1.3
            probabilities["extremely_short"] *= 0.5

        # If the conversation has been going on for a while, vary more
        if context.get("message_count", 0) > 5:
            probabilities["extremely_short"] *= 1.2
            probabilities["long"] *= 1.2

    def _adjust_probabilities_for_variety(self, probabilities: Dict[str, float]) -> None:
        """
        Adjust probabilities to avoid repetitive patterns

        Args:
            probabilities: The current probability distribution
        """
        # If we've had the same response type multiple times in a row, reduce its probability
        if self.consecutive_same_type_count > 0 and self.last_response_type:
            # More aggressive reduction to avoid repetition
            reduction_factor = min(0.3, 0.8 ** self.consecutive_same_type_count)
            probabilities[self.last_response_type] *= reduction_factor

            # Force a dramatic change in response length more frequently
            if self.consecutive_same_type_count >= 1 and random.random() < 0.8:
                # If we've been giving short responses, strongly favor longer ones
                if self.last_response_type in ["extremely_short", "slightly_short"]:
                    probabilities["slightly_long"] *= 3.0
                    probabilities["long"] *= 3.0
                    probabilities["medium"] *= 2.0
                # If we've been giving medium responses, favor extremes
                elif self.last_response_type == "medium":
                    probabilities["extremely_short"] *= 2.5
                    probabilities["long"] *= 2.5
                # If we've been giving long responses, strongly favor shorter ones
                elif self.last_response_type in ["slightly_long", "long"]:
                    probabilities["extremely_short"] *= 3.0
                    probabilities["slightly_short"] *= 3.0

            # Occasionally force a completely random response length
            if random.random() < 0.2:
                # Choose a random response type that's different from the last one
                response_types = list(probabilities.keys())
                response_types.remove(self.last_response_type)
                random_type = random.choice(response_types)
                # Boost its probability significantly
                probabilities[random_type] *= 4.0

    def _apply_randomness(self, probabilities: Dict[str, float]) -> None:
        """
        Apply randomness factor to probabilities

        Args:
            probabilities: The current probability distribution
        """
        randomness = config.RESPONSE_LENGTH_RANDOMNESS
        for key in probabilities:
            # Apply random adjustment within the randomness factor range
            random_adjustment = 1.0 + randomness * (random.random() * 2 - 1)
            probabilities[key] *= random_adjustment

    def _select_response_type(self, probabilities: Dict[str, float]) -> str:
        """
        Select a response type based on the probability distribution

        Args:
            probabilities: The normalized probability distribution

        Returns:
            Selected response type
        """
        # Convert probabilities to cumulative distribution
        items = list(probabilities.items())
        cumulative_prob = 0
        cumulative_probs = []

        for item, prob in items:
            cumulative_prob += prob
            cumulative_probs.append((item, cumulative_prob))

        # Select based on random value
        rand_val = random.random()
        for item, cum_prob in cumulative_probs:
            if rand_val <= cum_prob:
                return item

        # Fallback to medium if something goes wrong
        return "medium"

    def get_response_length_instructions(self, response_type: str) -> str:
        """
        Get specific instructions for the selected response length

        Args:
            response_type: The selected response type

        Returns:
            Instructions for the model to generate a response of the appropriate length
        """
        instructions = {
            "extremely_short": "Keep your response EXTREMELY SHORT - just a few words or a single short sentence. Be very concise.",
            "slightly_short": "Keep your response SLIGHTLY SHORT - use 1-2 concise sentences. Be brief but clear.",
            "medium": "Use a MEDIUM-LENGTH response - 2-3 sentences with moderate detail. Balance brevity and informativeness.",
            "slightly_long": "Use a SLIGHTLY LONG response - 3-5 sentences with good detail. Provide a thorough answer.",
            "long": "Give a LONG, detailed response - 5+ sentences with extensive detail. Be comprehensive and thorough."
        }

        return instructions.get(response_type, "Use a natural, conversational length for your response.")

    def format_response_length_for_prompt(self, message_content: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Format response length instructions for inclusion in the prompt

        Args:
            message_content: The user's message content
            context: Optional context information about the conversation

        Returns:
            Formatted response length instructions for the prompt
        """
        if not config.DYNAMIC_MESSAGE_LENGTH_ENABLED:
            return ""

        response_type = self.get_response_type(message_content, context)
        instructions = self.get_response_length_instructions(response_type)

        return f"""
        RESPONSE LENGTH INSTRUCTION:
        {instructions}
        """

# Create a singleton instance
dynamic_response_manager = DynamicResponseManager()
