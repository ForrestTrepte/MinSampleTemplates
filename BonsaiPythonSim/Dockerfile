# this is one of the cached base images available for ACI
FROM python:3.12

# Install libraries and dependencies
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  zlib1g-dev \
  swig

# Set up the simulator
WORKDIR /src

# Copy simulator files to /src
COPY . /src

# Install simulator dependencies
RUN python -m pip install --upgrade pip && \
  python -m pip install -r requirements.txt

# This will be the command to run the simulator
CMD ["python3", "main.py"]
