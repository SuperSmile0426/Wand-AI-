import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = "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"
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")
