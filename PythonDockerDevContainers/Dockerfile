FROM python:3.12-slim

# Workdir where Dev Containers will mount this git repo in the container
# Note that, after the container is built, WORKDIR will be overwritten with the repository content
WORKDIR /workspaces/MinSampleTemplates/PythonDockerDevContainers

# Install uv and package dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venvs/python-docker-devcontainers
COPY pyproject.toml uv.lock ./
RUN uv sync

CMD ["bash"]
