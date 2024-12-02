"""
Main entry point to get the instructor client
"""

from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel
from openai import OpenAI
import logging
import instructor
from instructor import Instructor
from typing import List, Tuple, Type, TypeAlias
import yaml

from get_vertexai import get_vertexai, get_project_id
from get_openai import get_openai

LLMClient: TypeAlias = OpenAI | GenerativeModel

with open("settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

tool_settings = settings["tool_settings"]
vertexai_settings = settings["vertexai"]
openai_settings = settings["openai"]

logger = logging.getLogger(__name__)
logger.setLevel(tool_settings["verbosity"])

logger.info("Initialized logger")

load_dotenv()
logger.info("Loaded enviromental variables")

# Initializing LLM services
if "vertexai" in tool_settings["service_priority"]:
    try:
        logger.info("Initializing vertexai")
        vertexai.init(project=get_project_id(), location=vertexai_settings["location"])
    except Exception as e:
        logger.warning(
            f"Failed to initialize vertexai due to the following Exception: \n {e}"
        )

service_to_getfunc_map = {"openai": get_openai, "vertexai": get_vertexai}

service_to_instructor_constructor = {
    OpenAI: instructor.from_openai,
    GenerativeModel: instructor.from_vertexai,
}


def get_client_by_priority(service_list: List[str]) -> LLMClient:
    """Obtains a supported LLM client (currently `OpenAI` or vertexai `GenerativeModel`) by priority list in config

    Args:
        service_list (List[str]): The priority list

    Raises:
        ClientInitializationError: raised when no client could be initialized from the provided service list.

    Returns:
        LLMClient: An instance of supported LLM client type
    """
    for service in service_list:
        logger.info(f"Initializing {service} client")
        client = service_to_getfunc_map[service]
        if client:
            return client
    logger.error("Cannot initialize any client from service list")
    raise ClientInitializationError(
        "Failed to initialize any client from the provided service list"
    )


def get_instructor() -> Tuple[Instructor, dict]:
    """Function to get an `Instructor` client (and corresponding inference arguments) according to the model

    Returns:
        Tuple[Instructor, dict]: The `Instructor` `dict` pair for inference use
    """
    llm_client = get_client_by_priority(tool_settings["service_priority"])
    logging.info("Obtained LLM client")
    inference_args = get_inference_args(type(llm_client))
    instructor_client = make_instructor_by_llm_client(llm_client)
    logging.info("Obtained Instructor Client")
    return instructor_client, inference_args


def get_inference_args(client_type: Type[LLMClient]) -> dict:
    """Get inference arguments based on the LLM client type

    Args:
        client_type (Type[LLMClient]): The type/class of the LLM client

    Returns:
        dict: Inference arguments for the specified client type
    """
    if client_type is OpenAI:
        return openai_settings["client_args"]
    elif client_type is GenerativeModel:
        return dict()
    else:
        logger.warning("Inference args not set for current LLM client type")
        return dict()


def make_instructor_by_llm_client(client: LLMClient):
    return service_to_instructor_constructor[type(client)](
        client, mode=instructor.Mode.JSON
    )


class ClientInitializationError(Exception):
    """Exception raised when no client could be initialized from the provided service list."""

    pass
