from openai import OpenAI
import logging
import yaml
from typing import Optional
import os

with open("settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

tool_settings = settings["tool_settings"]
openai_settings = settings["openai"]

logger = logging.getLogger(__name__)
logger.setLevel(tool_settings["verbosity"])


def get_openai() -> Optional[OpenAI]:
    """Helper function to obtain an `openAI` client using settings file `settings.yaml`

    Returns:
        Optional[OpenAI]: The returned client, returns None if cannot obtain a client
    """
    try:
        client = OpenAI(
            base_url=openai_settings["base_url"], api_key=os.getenv("OPENAI_API_KEY")
        )
    except Exception as e:
        logger.warning(
            f"Cannot initialize OpenAI client due to the following exception \n {e}"
        )
        client = None
    return client
