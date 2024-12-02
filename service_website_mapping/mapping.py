from clients.get_instructor import get_instructor
from structured_outputs import get_structured_outputs
from pydantic import BaseModel
from typing import List
from functools import partial

with open("service-website-mapping/prompt.txt", "r") as file:
    sys_prompt = file.read()

sys_msg = {"role": "system", "content": sys_prompt}


class SericeNameUrlPair(BaseModel):
    service: str
    url: str


structured_client, structured_client_args = get_instructor()

inference_func = partial(
    get_structured_outputs,
    response_model=SericeNameUrlPair,
    structured_client=structured_client,
    client_args=structured_client_args,
)


def map_service_to_og_url(service_name: str, candidates_urls: List[str]) -> dict:
    """Matches a service with the correct Open Graph Tag url using LLMs, settings can be changed in settings.yaml

    Args:
        service_name (str): The name of the service
        candidates_urls (List[str]): Candidate URLs for the service

    Returns:
        dict: A dictionary response containing the following keys:
              "service": The name of the service
              "url": The matched url
    """
    candidate_url_string_list = [f"{i}: {url}" for i, url in enumerate(candidates_urls)]
    user_prompt = "\n".join(
        [
            f"Candidate service: {service_name},",
            " candidate urls:",
            "\n".join(candidate_url_string_list),
        ]
    )
    user_msg = {"role": "user", "content": user_prompt}
    response: BaseModel = get_structured_outputs([sys_msg, user_msg])
    return response.model_dump()
