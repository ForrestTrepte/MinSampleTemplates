FROM python:3.14.2-slim

WORKDIR /workspaces/MinSampleTemplates/openai_reponses

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venvs/openai-reponses
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked

CMD ["bash"]
