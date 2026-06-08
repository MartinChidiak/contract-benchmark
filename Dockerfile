FROM nvidia/cuda:12.9.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# -------------------------
# System deps
# -------------------------
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev python3-venv \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Virtual env
# -------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# =========================================================
# Step 1 — vLLM first so pip resolves torch with CUDA support.
# Pinned to the version recorded in requirements-frozen.txt.
# =========================================================
RUN pip install --no-cache-dir vllm==0.22.1 accelerate==1.13.0

# =========================================================
# Step 2 — App stack at exact frozen versions.
# These have no CUDA deps so standard PyPI resolution is fine.
# =========================================================
RUN pip install --no-cache-dir \
    streamlit==1.58.0 \
    pandas==3.0.3 \
    matplotlib==3.10.9 \
    plotly==6.8.0 \
    scikit-learn==1.9.0

# =========================================================
# Step 3 — Lock remaining transitive deps to frozen versions.
# --no-deps: packages are already installed from steps 1+2;
# this just downgrades/upgrades any stragglers to match the freeze.
# =========================================================
COPY requirements-pinned.txt /tmp/requirements-pinned.txt
RUN pip install --no-cache-dir --no-deps -r /tmp/requirements-pinned.txt

# -------------------------
# User setup
# -------------------------
ARG HOST_UID=1000
ARG HOST_GID=1000
USER ${HOST_UID}:${HOST_GID}

WORKDIR /app

EXPOSE 8000 8501

CMD ["/bin/bash"]
