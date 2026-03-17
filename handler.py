"""RunPod Serverless handler: GPU-accelerated FFmpeg render.

Input:
  video_url: URL to download source video
  ass_content: ASS subtitle content (string)
  callback_url: URL to upload result back (PUT)

Flow:
  1. Download video from video_url
  2. Write ASS to temp file
  3. FFmpeg render with h264_nvenc (GPU)
  4. Upload result to callback_url
  5. Return output size and render time
"""
import os
import subprocess
import tempfile
import time

import requests
import runpod


def download_file(url: str, dest: str) -> int:
    """Download file from URL, return size in bytes."""
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    size = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            size += len(chunk)
    return size


def upload_file(path: str, url: str) -> None:
    """Upload file via PUT to callback URL."""
    with open(path, "rb") as f:
        r = requests.put(url, data=f, timeout=600,
                         headers={"Content-Type": "video/mp4"})
        r.raise_for_status()


def render_gpu(video_path: str, ass_path: str, output_path: str) -> float:
    """FFmpeg render with NVENC GPU encoding. Returns elapsed seconds."""
    escaped_ass = ass_path.replace(":", "\\:")

    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-i", video_path,
        "-vf", f"ass='{escaped_ass}'",
        "-c:v", "h264_nvenc",
        "-preset", "p4",       # balanced speed/quality
        "-cq", "23",           # constant quality mode
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ]

    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        # Fallback to CPU if NVENC not available (e.g. wrong GPU)
        print(f"NVENC failed, falling back to CPU: {result.stderr[-200:]}")
        cmd_cpu = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"ass='{escaped_ass}'",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "23",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
        result = subprocess.run(cmd_cpu, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr[-300:]}")

    return time.monotonic() - t0


def handler(event):
    """RunPod serverless handler."""
    inp = event.get("input", {})
    video_url = inp.get("video_url")
    ass_content = inp.get("ass_content")
    callback_url = inp.get("callback_url")

    if not video_url or not ass_content:
        return {"error": "video_url and ass_content required"}

    tmp = tempfile.mkdtemp(prefix="klipza_gpu_")
    video_path = os.path.join(tmp, "input.mp4")
    ass_path = os.path.join(tmp, "subs.ass")
    output_path = os.path.join(tmp, "output.mp4")

    try:
        # 1. Download video
        t0 = time.monotonic()
        video_size = download_file(video_url, video_path)
        dl_time = time.monotonic() - t0
        print(f"Downloaded {video_size / 1024 / 1024:.1f} MB in {dl_time:.1f}s")

        # 2. Write ASS
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_content)

        # 3. Render with GPU
        render_time = render_gpu(video_path, ass_path, output_path)
        output_size = os.path.getsize(output_path)
        print(f"Rendered in {render_time:.1f}s, output {output_size / 1024 / 1024:.1f} MB")

        # 4. Upload result
        if callback_url:
            t_up = time.monotonic()
            upload_file(output_path, callback_url)
            upload_time = time.monotonic() - t_up
            print(f"Uploaded in {upload_time:.1f}s")
        else:
            upload_time = 0

        return {
            "status": "done",
            "download_time": round(dl_time, 1),
            "render_time": round(render_time, 1),
            "upload_time": round(upload_time, 1),
            "total_time": round(time.monotonic() - t0, 1),
            "input_size_mb": round(video_size / 1024 / 1024, 1),
            "output_size_mb": round(output_size / 1024 / 1024, 1),
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Cleanup
        for f in [video_path, ass_path, output_path]:
            try:
                os.unlink(f)
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass


runpod.serverless.start({"handler": handler})
