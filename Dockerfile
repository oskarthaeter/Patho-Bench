FROM nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Build-time arguments for user id mapping
ARG USERNAME
ARG UID
ARG GID

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    curl \
    git \
    ca-certificates \
    libgl1 \
    python3.12-dev \
    python3-pip \
    python3-venv \
    tmux \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a virtual environment and install packages there
ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH
ENV PATH="${VENV_PATH}/bin:$PATH"
RUN python -m pip install --upgrade pip setuptools wheel

# Create a user and group with the specified UID and GID (tolerate pre-existing IDs)
RUN if ! getent group "$GID" >/dev/null; then \
        groupadd --gid "$GID" "$USERNAME"; \
    fi && \
    if id -u "$USERNAME" >/dev/null 2>&1; then \
        usermod --non-unique --uid "$UID" --gid "$GID" "$USERNAME"; \
    else \
        useradd --no-log-init --non-unique --uid "$UID" --gid "$GID" --create-home --shell /bin/bash "$USERNAME"; \
    fi && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R "$USERNAME:$GID" $VENV_PATH

# Install core PyTorch packages
RUN python -m pip install --no-cache-dir \
    torch==2.10.0+cu129 \
    torchvision==0.25.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129

# Install PyTorch ecosystem
RUN python -m pip install --no-cache-dir \
    torchdata \
    torchao==0.13.0 \
    triton \
    timm \
    huggingface_hub

RUN python -m pip install --no-cache-dir xformers==0.0.35 --extra-index-url https://download.pytorch.org/whl/cu129

RUN python -m pip install --no-cache-dir \
    lightning \
    torchmetrics \
    torch-tb-profiler \
    omegaconf

# Install I/O and image processing libraries
RUN python -m pip install --no-cache-dir \
    openslide-bin \
    openslide-python \
    h5py \
    pillow \
    tifffile \
    scikit-image

# Install experiment management and dev tools
RUN python -m pip install --no-cache-dir \
    tensorboard \
    wandb \
    python-dotenv \
    black \
    joblib \
    pandas \
    numpy \
    matplotlib \
    plotly \
    seaborn

# Install additional ML utilities
RUN python -m pip install --no-cache-dir \
    umap-learn \
    scikit-learn \
    flask \
    distinctipy \
    ipykernel

# Install Patho-Bench requirements
# trident requires --ignore-requires-python because it pins an older Python version
COPY requirements.txt .
RUN grep -v '^git+https://github.com/mahmoodlab/trident.git$' requirements.txt > /tmp/requirements-no-trident.txt && \
    python -m pip install --no-cache-dir --ignore-requires-python git+https://github.com/mahmoodlab/trident.git && \
    python -m pip install --no-cache-dir -r /tmp/requirements-no-trident.txt && \
    rm -f /tmp/requirements-no-trident.txt

# Clean pip cache
RUN rm -rf /root/.cache/pip

# Ensure the devcontainer remote user can update packages during postCreateCommand
RUN chown -R $USERNAME:$GID $VENV_PATH

ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

# Switch to the new user
USER $USERNAME

# Set the working directory
WORKDIR /workspaces

ENV PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync,expandable_segments:False,pinned_use_cuda_host_register:True,pinned_num_register_threads:4
ENV HUGGINGFACE_HUB_CACHE=/mnt/.cache
ENV TORCH_HOME=/mnt/.torchhome
ENV DEV_CWD=/workspaces/Patho-Bench
