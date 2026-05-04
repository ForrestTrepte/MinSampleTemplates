A simple Python program [simple.py](simple.py) that calls LiteLLM.

# Setup

* Run from vscode using `Dev Containers: Reopen in Container`.
* Once the container is open, use `Python: Select Interpreter` > Enter interpreter path... > /opt/venvs/python-docker-devcontainers/bin/python
* In terminal, set an environment variable to your API keys:
```
export OPENAI_API_KEY='sk-your-api-key-here'
export ANTHROPIC_API_KEY='sk-your-api-key-here'
```

# Execution

* Open simple.py and run using `Debug: Start Debugging`.
* Or, `uv run simple.py` from the terminal
