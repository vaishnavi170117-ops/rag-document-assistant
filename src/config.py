"""
config.py — reads settings from config.yaml and secrets from .env,
then shares them with the rest of the project.

Every other file imports 'config' from here, so all settings live in
ONE place. Change config.yaml, and the whole project sees the change.
"""

import os                          # lets us read environment variables (secrets)
from pathlib import Path           # a clean, safe way to handle file paths
import yaml                        # lets us read the config.yaml file
from dotenv import load_dotenv     # loads secrets from the .env file

# Load the .env file so its secrets become available.
# (Does nothing if .env doesn't exist yet — that's fine for now.)
load_dotenv()

# Figure out the project's root folder.
# __file__ = this file's location (src/config.py).
# .parent.parent = go up two levels → the main project folder.
ROOT = Path(__file__).resolve().parent.parent


class Config:
    """Holds all settings. Reads config.yaml once, then serves values on demand."""

    def __init__(self):
        # Open and read config.yaml into a Python dictionary.
        with open(ROOT / "config.yaml", "r") as f:
            self._cfg = yaml.safe_load(f)

        # Secrets come from .env (NEVER hardcoded here).
        # os.getenv reads an environment variable; returns None if missing.
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

    # --- Below: easy shortcuts so other files can write config.chunk_size, etc. ---
    # @property lets us access these like variables, not functions.

    @property
    def embedding_model(self):
        return self._cfg["embedding"]["model_name"]

    @property
    def chunk_size(self):
        return self._cfg["chunking"]["chunk_size"]

    @property
    def chunk_overlap(self):
        return self._cfg["chunking"]["chunk_overlap"]

    @property
    def top_k(self):
        return self._cfg["retrieval"]["top_k"]

    @property
    def llm_provider(self):
        return self._cfg["llm"]["provider"]

    @property
    def llm_model(self):
        return self._cfg["llm"]["model"]

    @property
    def temperature(self):
        return self._cfg["llm"]["temperature"]

    @property
    def max_tokens(self):
        return self._cfg["llm"]["max_tokens"]

    @property
    def raw_dir(self):
        return ROOT / self._cfg["paths"]["raw_dir"]

    @property
    def index_dir(self):
        return ROOT / self._cfg["paths"]["index_dir"]


# Create ONE shared Config object the whole project imports.
config = Config()