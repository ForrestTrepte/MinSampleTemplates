# OpenAI Responses sample

This sample calls the OpenAI Responses API using [simple.py](simple.py).

## Setup

1. Open this folder in VS Code and run `Dev Containers: Reopen in Container`.
2. Set your OpenAI API key in the container terminal:

   ```sh
   export OPENAI_API_KEY='your-api-key'
   ```

   If `OPENAI_API_KEY_PERSONAL` is set, the script uses it instead of
   `OPENAI_API_KEY`.

## Run

Run `uv run simple.py` in that terminal. You can also run the script with
VS Code's Python debugger; set the API key in the debugger's environment first.
