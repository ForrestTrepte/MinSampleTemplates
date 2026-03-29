A simple Python program [app.py](app.py) that can be run inside a Docker container usiing VSCode Dev Containers.

Python file manually from terminal:
1. Ctrl+Shift+P, Dev Containers: Reopen in Container
    * May need to switch back to local and then reopen container if terminal doesn't give you a prompt
2. In terminal `uv run python app.py`

Python file in debugger:
1. Set breakpoint in [app.py](app.py).
2. Ctrl+Shift+P, Dev Containers: Reopen in Container
    * May need to switch back to local and then reopen container if terminal doesn't give you a prompt
3. Ctrl+Shift+P, Python: Select Interpreter > Enter interpreter path... > /opt/venvs/python-docker-devcontainers/bin/python
4. F5 to start debugging (Start Debugging, Python Debugger: Current File)

Jupyter notebook:
1. Ctrl+Shift+P, Dev Containers: Reopen in Container
    * May need to switch back to local and then reopen container if terminal doesn't give you a prompt
2. Open jupyter_notebook.ipynb
3. Select kernel (upper right) > Select another kernel > Python environments > /opt/venvs/python-docker-devcontainers/bin/python
3. Run cell

What the files are doing:
* Dockerfile: How to build the docker container
* pyproject.toml: Python packages to install (via uv, invoked in the Dockerfile)
* devcontainer.json: How vscode Dev Containers will use the container
* launch.json: How to run the debugger
