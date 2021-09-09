# Bonsai Python Sim - Min Template

## Set up Python environment

Create a conda environment by:

```bash
conda create -n minbonsai python=3.7
conda activate minbonsai
pip install -r requirements.txt
```

## Connecting a local instance of the simulator to a brain

Run the simulator locally by:

```bash
python main.py
```

## Building Simulator Packages

Using the `azure-cli`, you can build the provided dockerfile to create a simulator package (note the trailing period!):

```azurecli
az acr build --image bonsai-python-sim:1 --file Dockerfile --registry <ACR_REGISTRY> .
```