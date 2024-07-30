import os
from langchain_openai import ChatOpenAI

# In order to make it easy to run work projects and personal AI experiments, override OPENAI_API_KEY with the value of OPENAI_API_KEY_PERSONAL if it is set.
if "OPENAI_API_KEY_PERSONAL" in os.environ:
    print("Using key from OPENAI_API_KEY_PERSONAL environment variable")
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY_PERSONAL"]

llm = ChatOpenAI(temperature=0.0, model_name="gpt-4o")  # type: ignore
result = llm.invoke(
    "What would be a good company name for a company that makes colorful socks?"
)
print(result.content)
