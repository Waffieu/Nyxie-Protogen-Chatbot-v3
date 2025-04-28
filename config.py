import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Bot configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Memory settings
SHORT_MEMORY_SIZE = int(os.getenv("SHORT_MEMORY_SIZE", "25"))
LONG_MEMORY_SIZE = int(os.getenv("LONG_MEMORY_SIZE", "100"))
MEMORY_DIR = os.getenv("MEMORY_DIR", "user_memories")

# Web search settings
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "100"))

# Proxy settings - DISABLED
# Proxy system has been removed due to connection issues with DuckDuckGo
PROXY_ENABLED = False
PROXY_LIST = []
PROXY_FILE = ""

# Maximum number of retries for DuckDuckGo searches
MAX_SEARCH_RETRIES = int(os.getenv("MAX_SEARCH_RETRIES", "10"))

# Time awareness settings
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "Europe/Istanbul")
TIME_AWARENESS_ENABLED = os.getenv("TIME_AWARENESS_ENABLED", "true").lower() == "true"
# Only show time information when relevant to the conversation
SHOW_TIME_ONLY_WHEN_RELEVANT = os.getenv("SHOW_TIME_ONLY_WHEN_RELEVANT", "true").lower() == "true"

# Website link settings
# Only show website links when explicitly requested or relevant
SHOW_LINKS_ONLY_WHEN_RELEVANT = os.getenv("SHOW_LINKS_ONLY_WHEN_RELEVANT", "true").lower() == "true"

# Self-awareness and environmental awareness settings
SELF_AWARENESS_ENABLED = os.getenv("SELF_AWARENESS_ENABLED", "true").lower() == "true"
ENVIRONMENT_AWARENESS_ENABLED = os.getenv("ENVIRONMENT_AWARENESS_ENABLED", "true").lower() == "true"
SELF_AWARENESS_SEARCH_ENABLED = os.getenv("SELF_AWARENESS_SEARCH_ENABLED", "true").lower() == "true"
# Level of detail for environmental awareness (1-5)
ENVIRONMENT_AWARENESS_LEVEL = int(os.getenv("ENVIRONMENT_AWARENESS_LEVEL", "3"))

# Word translation settings
WORD_TRANSLATION_ENABLED = os.getenv("WORD_TRANSLATION_ENABLED", "true").lower() == "true"
# Minimum word length to consider for translation
MIN_WORD_LENGTH_FOR_TRANSLATION = int(os.getenv("MIN_WORD_LENGTH_FOR_TRANSLATION", "4"))
# Maximum number of words to translate per message
MAX_WORDS_TO_TRANSLATE = int(os.getenv("MAX_WORDS_TO_TRANSLATE", "5"))

# Dynamic message length settings
DYNAMIC_MESSAGE_LENGTH_ENABLED = os.getenv("DYNAMIC_MESSAGE_LENGTH_ENABLED", "true").lower() == "true"
# Probability distribution for different response lengths
# These values determine the approximate probability of each response type
# More balanced distribution to ensure variety
EXTREMELY_SHORT_RESPONSE_PROBABILITY = float(os.getenv("EXTREMELY_SHORT_RESPONSE_PROBABILITY", "0.15"))
SLIGHTLY_SHORT_RESPONSE_PROBABILITY = float(os.getenv("SLIGHTLY_SHORT_RESPONSE_PROBABILITY", "0.20"))
MEDIUM_RESPONSE_PROBABILITY = float(os.getenv("MEDIUM_RESPONSE_PROBABILITY", "0.25"))
SLIGHTLY_LONG_RESPONSE_PROBABILITY = float(os.getenv("SLIGHTLY_LONG_RESPONSE_PROBABILITY", "0.20"))
LONG_RESPONSE_PROBABILITY = float(os.getenv("LONG_RESPONSE_PROBABILITY", "0.20"))
# Randomness factor for response length (0.0-1.0, higher = more random)
RESPONSE_LENGTH_RANDOMNESS = float(os.getenv("RESPONSE_LENGTH_RANDOMNESS", "0.9"))

# Slang and casual language settings
SLANG_ENABLED = os.getenv("SLANG_ENABLED", "true").lower() == "true"
# Probability of using slang in a response (0.0-1.0)
SLANG_PROBABILITY = float(os.getenv("SLANG_PROBABILITY", "0.4"))
# Maximum level of slang/swearing (1-5, where 5 is most casual/explicit)
SLANG_LEVEL = int(os.getenv("SLANG_LEVEL", "3"))

# Gemini model settings
GEMINI_MODEL = "gemini-2.5-pro-exp-03-25"
GEMINI_TEMPERATURE = 0.8  # Slightly higher temperature for more varied, natural responses
GEMINI_TOP_P = 0.95
GEMINI_TOP_K = 40
GEMINI_MAX_OUTPUT_TOKENS = 2048  # Reduced max tokens to encourage shorter responses

# Specialized Gemini models
# Model for web search and language detection
GEMINI_FLASH_LITE_MODEL = "gemini-2.0-flash-lite"
GEMINI_FLASH_LITE_TEMPERATURE = 0.4
GEMINI_FLASH_LITE_TOP_P = 0.95
GEMINI_FLASH_LITE_TOP_K = 32
GEMINI_FLASH_LITE_MAX_OUTPUT_TOKENS = 8192

# Model for image analysis
GEMINI_IMAGE_MODEL = "gemini-2.5-pro-exp-03-25"
GEMINI_IMAGE_TEMPERATURE = 0.7
GEMINI_IMAGE_TOP_P = 0.95
GEMINI_IMAGE_TOP_K = 40
GEMINI_IMAGE_MAX_OUTPUT_TOKENS = 4096

# Model for word translation
GEMINI_TRANSLATION_MODEL = "gemini-2.0-flash-lite"
GEMINI_TRANSLATION_TEMPERATURE = 0.1
GEMINI_TRANSLATION_TOP_P = 0.95
GEMINI_TRANSLATION_TOP_K = 40
GEMINI_TRANSLATION_MAX_OUTPUT_TOKENS = 1024

# Safety settings - all set to BLOCK_NONE as requested
SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
]
