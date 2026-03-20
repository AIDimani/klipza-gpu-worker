FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps (libass for subtitle filter, fontconfig for fc-cache)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libass9 fontconfig curl python3 python3-pip xz-utils ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# BtbN static FFmpeg — lgpl build includes NVENC via nv-codec-headers
# If NVENC missing, switch to nonfree variant
# Verify exact filename: curl -s https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest | jq '.assets[].name'
RUN curl -L https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-nonfree.tar.xz \
    -o /tmp/ffmpeg.tar.xz \
    && tar -xJf /tmp/ffmpeg.tar.xz --strip-components=2 -C /usr/local/bin/ \
       ffmpeg-master-latest-linux64-nonfree/bin/ffmpeg \
       ffmpeg-master-latest-linux64-nonfree/bin/ffprobe \
    && rm /tmp/ffmpeg.tar.xz \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe

# Klipza subtitle fonts
COPY fonts/ /usr/local/share/fonts/
RUN fc-cache -f -v

# Python deps
RUN pip3 install --no-cache-dir runpod==1.7.* requests

WORKDIR /app
COPY handler.py .

# MUST fail build if NVENC or ASS missing
RUN ffmpeg -encoders 2>/dev/null | grep -q h264_nvenc || (echo "ERROR: h264_nvenc not found — use nonfree build" && exit 1)
RUN ffmpeg -filters 2>/dev/null | grep -q "ass" || (echo "ERROR: ass filter not found" && exit 1)

CMD ["python3", "-u", "handler.py"]
