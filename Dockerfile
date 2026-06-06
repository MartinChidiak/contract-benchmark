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
# vLLM — latest version, includes sm_120 (RTX 5090) support
# No version pin: whatever pip resolves is recorded in the freeze below
# =========================================================
RUN pip install --no-cache-dir vllm accelerate

# =========================================================
# App stack
# =========================================================
RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    matplotlib \
    plotly \
    scikit-learn

# =========================================================
# Hard freeze for reproducibility
# =========================================================
RUN pip freeze > /opt/requirements.lock.txt

# -------------------------
# User setup
# -------------------------
ARG HOST_UID=1000
ARG HOST_GID=1000
USER ${HOST_UID}:${HOST_GID}

WORKDIR /app

EXPOSE 8000 8501

CMD ["/bin/bash"]
