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

Build the provided dockerfile and upload it to your container registry.

```azurecli
docker build -t bonsai-python-sim:latest -f Dockerfile .
docker tag bonsai-python-sim:latest <ACR_REGISTRY_NAME>.azurecr.io/bonsai-python-sim:latest
docker push <ACR_REGISTRY_NAME>.azurecr.io/bonsai-python-sim:latest
```
