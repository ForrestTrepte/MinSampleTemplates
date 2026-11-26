# LangChain

Open this folder in VS Code and run **Dev Containers: Reopen in Container** to install the dependencies from `Pipfile.lock` and work in the Python 3.12 environment.

Set any required API keys in the dev container's terminal before running examples. For example, `simple.py` uses `OPENAI_API_KEY` (or `OPENAI_API_KEY_PERSONAL`).

## Updating package versions

```
pipenv update
pipenv audit --locked
```

## Environment variables

For the Jupyter notebook to pick up your keys, create a .env file (note that it is .gitignore'ed) like this:

``
OPENAI_API_KEY_PERSONAL=sk-...
ANTHROPIC_API_KEY_PERSONAL=sk-...
GEMINI_API_KEY_PERSONAL=...
``
