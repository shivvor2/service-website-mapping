from pydantic import BaseModel
from typing import List
from instructor import Instructor


def get_structured_outputs(
    messages: List[dict],
    response_model: BaseModel,
    structured_client: Instructor,
    client_args: dict,
):
    """Creates structured chat completions using an Instructor client with a specified response model.

    Args:
        messages (List[dict]): List of message dictionaries containing the conversation history
        response_model (BaseModel): Pydantic model class defining the expected response structure
        client_structured (Instructor): Initialized Instructor client instance
        client_args (dict): Additional arguments to pass to the chat completion creation

    Returns:
        BaseModel: A response object matching the specified response_model structure
    """
    response: BaseModel = structured_client.chat.completions.create(
        messages=messages, response_model=response_model, **client_args
    )
    return response
