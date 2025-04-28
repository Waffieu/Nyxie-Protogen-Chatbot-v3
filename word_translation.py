import logging
import re
import google.generativeai as genai
from typing import List, Dict, Any, Tuple
import config

# Configure logging
logger = logging.getLogger(__name__)

class WordTranslator:
    """
    Class to handle translation of uncommon words to Turkish
    """
    def __init__(self):
        """Initialize the word translator"""
        self.translation_cache = {}  # Cache for previously translated words
        self.uncommon_word_pattern = re.compile(r'\b[a-zA-ZçğıöşüÇĞİÖŞÜ]{4,}\b')  # Words with 4+ characters
        logger.info("Word translator initialized")

    def detect_uncommon_words(self, text: str, language: str) -> List[str]:
        """
        Detect potentially uncommon words in the text

        Args:
            text: The text to analyze
            language: The detected language of the text

        Returns:
            List of potentially uncommon words
        """
        # If the text is already in Turkish, no need to translate
        if language.lower() == "turkish":
            return []

        # Extract all words with 4+ characters
        all_words = self.uncommon_word_pattern.findall(text)

        # Filter out already translated words and proper nouns
        candidate_words = []
        for word in all_words:
            word_lower = word.lower()

            # Skip if the word is already in our cache
            if word_lower in self.translation_cache:
                continue

            # Skip words that are likely proper nouns (capitalized in the middle of a sentence)
            if word[0].isupper() and not word_lower.isupper():
                # Check if it's not at the beginning of a sentence
                word_index = text.find(word)
                if word_index > 0 and text[word_index-1] not in ".!?\n":
                    continue

            # Add to the list of candidate words
            candidate_words.append(word)

        # We'll let Gemini determine the level of each word during translation
        # Just return a reasonable number of candidates
        return candidate_words[:10]  # Increased limit to get more candidates

    async def translate_uncommon_words(self, text: str, language: str) -> Tuple[str, Dict[str, str]]:
        """
        Detect and translate uncommon words in the text

        Args:
            text: The text to process
            language: The detected language of the text

        Returns:
            Tuple of (processed text, dictionary of translations)
        """
        try:
            # If the text is already in Turkish, no need to translate
            if language.lower() == "turkish":
                return text, {}

            # First try to detect uncommon words using our main algorithm
            uncommon_words = self.detect_uncommon_words(text, language)

            # If we found fewer than 3 words, try to find more using additional methods
            if len(uncommon_words) < 3:
                # Extract words with 5+ characters
                longer_words = re.findall(r'\b[a-zA-ZçğıöşüÇĞİÖŞÜ]{5,}\b', text)

                # Filter out words that are already in our uncommon_words list or in our cache
                longer_words = [word for word in longer_words
                               if word.lower() not in [w.lower() for w in uncommon_words]
                               and word.lower() not in self.translation_cache]

                # Add more words until we have at least 3 (or as many as we can find)
                needed_words = 3 - len(uncommon_words)
                if longer_words:
                    uncommon_words.extend(longer_words[:needed_words])

            # If we still don't have enough words, add some predefined A2+ level words
            # that are likely to appear in conversations
            predefined_a2_words = [
                "serendipity", "ephemeral", "ubiquitous", "eloquent", "meticulous",
                "ambiguous", "pragmatic", "resilient", "profound", "intricate",
                "exquisite", "quintessential", "juxtaposition", "melancholy", "nostalgia",
                "euphoria", "tranquility", "serene", "enigmatic", "sophisticated",
                "meticulous", "diligent", "comprehensive", "elaborate", "substantial",
                "significant", "considerable", "extensive", "tremendous", "extraordinary"
            ]

            # If we still need more words, add some from our predefined list
            if len(uncommon_words) < 3:
                # Shuffle the predefined words to get random ones each time
                import random
                random.shuffle(predefined_a2_words)

                # Add words until we have at least 3
                needed_words = 3 - len(uncommon_words)
                for word in predefined_a2_words[:needed_words]:
                    if word.lower() not in [w.lower() for w in uncommon_words]:
                        uncommon_words.append(word)

            logger.info(f"Detected/selected {len(uncommon_words)} words for translation")

            # Translate the uncommon words
            translations = await self._get_translations(uncommon_words, language)

            # Update the cache with new translations
            self.translation_cache.update(translations)

            # Return the original text and the translations
            return text, translations

        except Exception as e:
            logger.error(f"Error translating uncommon words: {e}")
            return text, {}

    async def translate_uncommon_words_in_text(self, text: str, words_list: List[str], language: str) -> Tuple[str, Dict[str, str]]:
        """
        Translate specific words that appear in the text

        Args:
            text: The text to process
            words_list: List of words to consider for translation
            language: The detected language of the text

        Returns:
            Tuple of (processed text, dictionary of translations)
        """
        try:
            # If the text is already in Turkish, no need to translate
            if language.lower() == "turkish":
                return text, {}

            # Filter out words that are likely proper nouns or already in our cache
            candidate_words = []
            cached_translations = {}

            for word in words_list:
                word_lower = word.lower()

                # If the word is already in our cache, use the cached translation
                if word_lower in self.translation_cache:
                    cached_translations[word_lower] = self.translation_cache[word_lower]
                    continue

                # Skip words that are likely proper nouns (capitalized in the middle of a sentence)
                if word[0].isupper() and not word_lower.isupper():
                    # Check if it's not at the beginning of a sentence
                    word_index = text.find(word)
                    if word_index > 0 and text[word_index-1] not in ".!?\n":
                        continue

                # Add to the list of candidate words
                candidate_words.append(word)

            logger.info(f"Found {len(candidate_words)} new candidate words and {len(cached_translations)} cached translations")

            # Limit to a reasonable number of candidates
            if len(candidate_words) > 10:
                # Prioritize longer words as they're more likely to be uncommon
                candidate_words.sort(key=len, reverse=True)
                candidate_words = candidate_words[:10]

            # Translate the candidate words
            new_translations = await self._get_translations(candidate_words, language)

            # Update the cache with new translations
            self.translation_cache.update(new_translations)

            # Combine cached and new translations
            all_translations = {**cached_translations, **new_translations}

            # Return the original text and the translations
            return text, all_translations

        except Exception as e:
            logger.error(f"Error translating specific words in text: {e}")
            return text, {}

    async def force_translate_words(self, words_list: List[str], language: str) -> Tuple[str, Dict[str, str]]:
        """
        Force translate words regardless of their level

        Args:
            words_list: List of words to translate
            language: The detected language of the words

        Returns:
            Tuple of (empty string, dictionary of translations)
        """
        try:
            # If no words or Turkish language, return empty
            if not words_list or language.lower() == "turkish":
                return "", {}

            # Filter out words already in our cache
            new_words = []
            cached_translations = {}

            for word in words_list:
                word_lower = word.lower()

                # If the word is already in our cache, use the cached translation
                if word_lower in self.translation_cache:
                    cached_translations[word_lower] = self.translation_cache[word_lower]
                    continue

                # Add to the list of new words
                new_words.append(word)

            logger.info(f"Forcing translation of {len(new_words)} words")

            # Create a special prompt for Gemini to translate these words regardless of level
            prompt = f"""
            You are a professional linguist specializing in translating words from {language} to Turkish.

            For each of the following {language} words:
            1. Determine the CEFR level (A1, A2, B1, B2, C1, C2)
            2. Provide the Turkish translation
            3. IMPORTANT: Translate ALL words regardless of their level

            Words to translate:
            {", ".join(new_words)}

            Format your response exactly like this example:
            serendipity = C2 = tesadüf
            ephemeral = C1 = geçici
            ubiquitous = C1 = her yerde bulunan
            comfortable = A2 = rahat
            good = A1 = iyi
            """

            # Use Gemini to translate the words
            model = genai.GenerativeModel(
                model_name=config.GEMINI_TRANSLATION_MODEL,
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                },
                safety_settings=config.SAFETY_SETTINGS
            )

            response = model.generate_content(prompt)

            # Parse the response
            forced_translations = {}

            for line in response.text.strip().split('\n'):
                if '=' in line:
                    # Format should be: word = CEFR_LEVEL = translation
                    parts = line.split('=', 2)

                    if len(parts) >= 3:  # We have CEFR level and translation
                        word = parts[0].strip().lower()
                        cefr_level = parts[1].strip().upper()
                        translation = parts[2].strip()

                        # Only add translations for A2 and above levels
                        if cefr_level not in ["A1", "[A1]"]:
                            forced_translations[word] = f"{translation} ({cefr_level})"
                    elif len(parts) == 2:  # Fallback for old format
                        word = parts[0].strip().lower()
                        translation = parts[1].strip()
                        # Only add if it's not marked to be skipped or A1 level
                        if translation != "[SKIP]" and translation != "[A1]" and translation != "[COMMON]":
                            forced_translations[word] = translation

            # Update the cache with new translations
            self.translation_cache.update(forced_translations)

            # Combine cached and forced translations
            all_translations = {**cached_translations, **forced_translations}

            return "", all_translations

        except Exception as e:
            logger.error(f"Error force translating words: {e}")
            return "", {}

    async def _get_translations(self, words: List[str], source_language: str) -> Dict[str, str]:
        """
        Get translations for a list of words using Gemini

        Args:
            words: List of words to translate
            source_language: Source language of the words

        Returns:
            Dictionary mapping words to their Turkish translations
        """
        if not words:
            return {}

        try:
            # Create a prompt for Gemini to translate the words and determine their CEFR level
            # IMPORTANT: We're now translating ALL words regardless of level
            prompt = f"""
            You are a professional linguist specializing in translating words from {source_language} to Turkish.

            For each of the following {source_language} words:
            1. Determine the CEFR level (A1, A2, B1, B2, C1, C2)
            2. Provide the Turkish translation
            3. IMPORTANT: Translate ALL words regardless of their CEFR level
            4. Only skip proper nouns or words that are already Turkish

            CEFR LEVEL GUIDELINES:
            - A1 (Beginner): Very basic, everyday words that beginners learn first (e.g., "good", "house", "water")
            - A2 (Elementary): Common words used in everyday situations (e.g., "comfortable", "improve", "suggest")
            - B1 (Intermediate): More abstract words and less common everyday vocabulary (e.g., "achievement", "influence", "solution")
            - B2 (Upper Intermediate): More specialized vocabulary and abstract concepts (e.g., "controversy", "perspective", "sustainable")
            - C1 (Advanced): Sophisticated vocabulary, idioms, and specialized terms (e.g., "ambiguous", "meticulous", "pragmatic")
            - C2 (Proficiency): Very rare words, highly specialized terms (e.g., "ephemeral", "ubiquitous", "quintessential")

            Words to translate:
            {", ".join(words)}

            Format your response exactly like this example:
            serendipity = C2 = tesadüf
            ephemeral = C1 = geçici
            ubiquitous = C1 = her yerde bulunan
            comfortable = A2 = rahat
            good = A1 = iyi
            John = [SKIP]
            """

            # Use Gemini to translate the words
            model = genai.GenerativeModel(
                model_name=config.GEMINI_TRANSLATION_MODEL,
                generation_config={
                    "temperature": 0.2,  # Slightly higher temperature for more natural translations
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                },
                safety_settings=config.SAFETY_SETTINGS
            )

            response = model.generate_content(prompt)

            # Parse the response
            translations = {}
            cefr_levels = {}

            for line in response.text.strip().split('\n'):
                if '=' in line:
                    # Format should be: word = CEFR_LEVEL = translation
                    parts = line.split('=', 2)

                    if len(parts) >= 3:  # We have CEFR level and translation
                        word = parts[0].strip().lower()
                        cefr_level = parts[1].strip().upper()
                        translation = parts[2].strip()

                        # Store the CEFR level
                        cefr_levels[word] = cefr_level

                        # Only add translations for A2 and above levels
                        if translation != "[SKIP]" and cefr_level not in ["A1", "[A1]"]:
                            # Add CEFR level to the translation for display
                            translations[word] = f"{translation} ({cefr_level})"
                    elif len(parts) == 2:  # Fallback for old format
                        word = parts[0].strip().lower()
                        translation = parts[1].strip()

                        # Only add if it's not marked to be skipped or A1 level
                        if translation != "[SKIP]" and translation != "[A1]" and translation != "[COMMON]":
                            translations[word] = translation

            # Log the CEFR levels for debugging
            if cefr_levels:
                logger.info(f"CEFR levels: {cefr_levels}")

            logger.info(f"Translated {len(translations)} words")
            return translations

        except Exception as e:
            logger.error(f"Error getting translations: {e}")
            return {}

    def format_translations_for_response(self, translations: Dict[str, str]) -> str:
        """
        Format translations for inclusion in the response

        Args:
            translations: Dictionary of word translations

        Returns:
            Formatted string with translations
        """
        if not translations:
            return ""

        # Always use the multiple words format for consistency, even if there's only one translation
        # This makes it easier for the model to recognize and include translations
        translation_text = "\n\nKelime Çevirileri:\n"

        # Sort translations alphabetically for consistency
        sorted_translations = sorted(translations.items())

        for word, translation in sorted_translations:
            # Check if the translation already includes the CEFR level
            if "(" in translation and ")" in translation and any(level in translation for level in ["A1", "A2", "B1", "B2", "C1", "C2"]):
                # Translation already has CEFR level, use as is
                translation_text += f"• {word} = {translation}\n"
            else:
                # Add a default level if not present (should not happen with new system)
                translation_text += f"• {word} = {translation} (A2+)\n"

        return translation_text

# Create a singleton instance
word_translator = WordTranslator()
