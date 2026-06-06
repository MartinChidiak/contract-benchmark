# Usa la imagen de desarrollo para asegurar compatibilidad con la arquitectura de la 5090
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

# Evitar interacciones durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias de sistema
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear un entorno virtual para evitar el uso de --break-system-packages
# Esto es más seguro y estándar en Ubuntu 24.04
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Actualizar pip e instalar dependencias core
# vLLM 0.7.0+ requiere dependencias específicas de procesamiento de esquemas
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    vllm>=0.7.0 \
    outlines \
    pydantic \
    openai \
    accelerate \
    streamlit \
    pandas \
    matplotlib \
    plotly \
    scikit-learn

# Correr como usuario no-root con el mismo UID/GID del host WSL.
# Los archivos del venv son world-readable (755/644) — no requieren chown.
# Pasar en docker run: --build-arg HOST_UID=$(id -u) --build-arg HOST_GID=$(id -g)
ARG HOST_UID=1000
ARG HOST_GID=1000
USER ${HOST_UID}:${HOST_GID}

# Directorio de trabajo
WORKDIR /app

# Exponer puerto para el servidor API
EXPOSE 8000 8501

# Comando por defecto
CMD ["/bin/bash"]