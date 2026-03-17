FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# FFmpeg with NVENC support
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3 python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# RunPod SDK
RUN pip3 install --no-cache-dir runpod requests

WORKDIR /app
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
