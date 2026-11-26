FROM python:3.12-slim

WORKDIR /workspaces/MinSampleTemplates/LangChain

RUN pip install --no-cache-dir pipenv
COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy --ignore-pipfile

CMD ["bash"]
