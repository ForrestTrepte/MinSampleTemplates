import os

from openai import OpenAI

# In order to make it easy to run work projects and personal AI experiments, override OPENAI_API_KEY with the value of OPENAI_API_KEY_PERSONAL if it is set.
if "OPENAI_API_KEY_PERSONAL" in os.environ:
    print("Using key from OPENAI_API_KEY_PERSONAL environment variable")
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY_PERSONAL"]

client = OpenAI()
print(f"Using OpenAI client version {client._version}")
response = client.responses.create(
    model="gpt-5.2-2025-12-11",
    instructions="You are a marketing expert.",
    input="What would be a good company name for a company that makes colorful socks?",
)
print(response.output_text)
