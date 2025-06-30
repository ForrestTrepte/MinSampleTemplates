import os

import typing_extensions as typing
import vertexai
from langchain.globals import set_debug, set_verbose
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field
from vertexai.generative_models import GenerationConfig, GenerativeModel

method = "langchain structured vertexai"
# method = "langchain vertexai"

# In order to make it easy to run work projects and personal AI experiments, override these key values with the value of *_PERSONAL, if set.
if method in ["langchain vertexai", "google vertexai"]:
    if "GOOGLE_APPLICATION_CREDENTIALS_PERSONAL" in os.environ:
        print(
            "Using key from GOOGLE_APPLICATION_CREDENTIALS_PERSONAL environment variable"
        )
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ[
            "GOOGLE_APPLICATION_CREDENTIALS_PERSONAL"
        ]
elif method in ["langchain genai", "google genai"]:
    if "GOOGLE_API_KEY_PERSONAL" in os.environ:
        print("Using key from GOOGLE_API_KEY_PERSONAL environment variable")
        os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY_PERSONAL"]

response_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "recipe_name": {
                "type": "string",
            },
        },
        "required": ["recipe_name"],
    },
}


response_schema_object = {
    "title": "recipes",
    "description": "An object containing a list of recipes.",
    "type": "object",
    "properties": {
        "recipes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "recipe_name": {
                        "type": "string",
                    },
                },
                "required": ["recipe_name"],
            },
        },
    },
    "required": ["recipes"],
}


class RecipeTypedDict(typing.TypedDict):
    recipe_name: str


class RecipePydantic(BaseModel):
    recipe_name: str = Field(description="The name of the recipe")


class RecipesPydantic(BaseModel):
    recipes: list[RecipePydantic] = Field(description="List of recipes")


if method == "langchain vertexai":
    llm = ChatVertexAI(
        temperature=0.0,
        model_name="gemini-2.5-flash",
        response_mime_type="application/json",
        response_schema=response_schema,
        location="global",
    )  # type: ignore
    result = llm.invoke("List a few popular cookie recipes")
    print(result.content)
elif method == "langchain structured vertexai":
    # NOTE: Internally, this uses a tool instead of a JSON schema for output.
    llm = ChatVertexAI(
        temperature=0.0,
        model_name="gemini-2.5-flash",
        location="global",
    )  # type: ignore

    structured_llm = llm.with_structured_output(response_schema_object)
    # structured_llm = llm.with_structured_output(RecipesPydantic)

    result = structured_llm.invoke("List a few popular cookie recipes")
    print(result)
elif method == "langchain genai":
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        response_mime_type="application/json",
        response_schema=response_schema,
    )
    result = llm.invoke("List a few popular cookie recipes")
    print(result.content)
elif method == "google vertexai":
    vertexai.init(project="forrest-ai-sandbox", location="global")
    llm = GenerativeModel("gemini-2.5-flash")
    response = llm.generate_content(
        "List a few popular cookie recipes",
        generation_config=GenerationConfig(
            response_mime_type="application/json", response_schema=response_schema
        ),
    )
    print(response.text)
elif method == "google genai":
    print("google genai method not implemented")
    pass
