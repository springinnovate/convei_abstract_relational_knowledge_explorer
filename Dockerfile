#docker build -f Dockerfile -t research-embeddings . && docker run --rm -it --gpus all -v %CD%:/work research-embeddings
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    pkg-config \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libstdc++6 \
    libffi-dev \
    libssl-dev \
    libxml2 \
    libxslt1.1 \
    zlib1g \
    libz-dev \
    libblas3 \
    liblapack3 \
    gfortran \
    libopenblas0 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# GPU-enabled PyTorch (CUDA 12.1 wheels)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Sentence embeddings
RUN pip install --no-cache-dir \
    sentence-transformers

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    tqdm \
    python-igraph \
    leidenalg \
    sqlalchemy \
    faiss-cpu \
    python-dotenv \
    openai \
    tiktoken \
    networkx \
    matplotlib

WORKDIR /work

CMD ["bash"]
