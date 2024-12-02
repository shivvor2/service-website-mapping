import os
from vertexai.generative_models import GenerativeModel
import logging
import yaml
import json
from typing import Optional

with open("settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

tool_settings = settings["tool_settings"]
vertexai_settings = settings["vertexai"]
vertexai_gen_config = vertexai_settings["gen_config"]

logger = logging.getLogger(__name__)
logger.setLevel(tool_settings["verbosity"])


def get_vertexai() -> Optional[GenerativeModel]:
    """Helper function to obtain a VertexAI `GenerativeModel` client using settings file `settings.yaml`

    Returns:
        Optional[GenerativeModel]: The returned client, returns None if cannot obtain a client
    """
    try:
        client = GenerativeModel(
            model=vertexai_settings["model"],
            generation_config=vertexai_gen_config["gen_config"],
        )
    except Exception as e:
        logger.warning(
            f"Cannot initialize VertexAI client due to the following exception \n {e}"
        )
        client = None
    return client


def get_project_id() -> Optional[str]:
    """Gets the project_id from the application credential json file"""
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path is None:
        logger.warning("No google application credential found in environment")
        return None
    with open(credentials_path, "r") as f:
        credentials = json.load(f)
    project_id = credentials["project_id"]
    return project_id
