FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libass9 fontconfig curl python3 python3-pip xz-utils ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# BtbN GPL FFmpeg static build — includes h264_nvenc + libx264 + libass
RUN curl -L https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz \
    -o /tmp/ffmpeg.tar.xz \
    && cd /tmp && tar -xJf ffmpeg.tar.xz \
    && cp /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ffmpeg \
    && cp /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffprobe /usr/local/bin/ffprobe \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg*

# Klipza subtitle fonts
COPY fonts/ /usr/local/share/fonts/
RUN fc-cache -f -v

# Python deps
RUN pip3 install --no-cache-dir runpod==1.7.* requests

WORKDIR /app
COPY handler.py .

# Verify libx264 is available (does not need GPU, always works at build time)
RUN ffmpeg -encoders 2>/dev/null | grep -q libx264 || (echo "ERROR: libx264 not found in FFmpeg" && exit 1)
# Verify ASS subtitle filter
RUN ffmpeg -filters 2>/dev/null | grep -q "ass" || (echo "ERROR: ass filter not found" && exit 1)
# Note: h264_nvenc check skipped at build time (no GPU on build server). Checked at runtime by handler.

CMD ["python3", "-u", "handler.py"]
