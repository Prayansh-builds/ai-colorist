"""video.py - read a video frame by frame, write frames back, and keep the audio."""
import os, json, time, subprocess
import numpy as np

def _probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                        "-of", "json", path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffprobe could not read the file: " + r.stderr[:200])
    return json.loads(r.stdout)

def video_info(path):
    info = _probe(path)
    v = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    if v is None:
        raise RuntimeError("no video stream found in " + path)
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    if "rotate" in v.get("tags", {}):
        rot = int(float(v["tags"]["rotate"]))
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if abs(rot) in (90, 270):          # phone videos are often stored sideways
        w, h = h, w
    num, den = v["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    duration = float(v.get("duration") or info["format"]["duration"])
    return dict(width=w, height=h, fps=fps, duration=duration,
                frames=int(round(duration * fps)),
                has_audio=any(s.get("codec_type") == "audio" for s in info["streams"]))

def _even(n):
    return max(2, int(n) // 2 * 2)

def target_size(info, max_side=None):
    w, h = info["width"], info["height"]
    if max_side and max(w, h) > max_side:
        s = max_side / max(w, h)
        w, h = _even(w * s), _even(h * s)
    return w, h

def read_frames(path, max_side=None):
    # Gives you the frames one at a time, as pictures of numbers from 0 to 255
    info = video_info(path)
    w, h = target_size(info, max_side)
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.wait()

class VideoWriter:
    # Collects frames and binds them into an MP4 that plays everywhere
    def __init__(self, path, width, height, fps, crf=18):
        self.size = (height, width)
        self.p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
             "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE)

    def write(self, frame):
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
        if frame.shape[:2] != self.size:
            raise ValueError(f"frame is {frame.shape[:2]}, expected {self.size}")
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()

def add_audio(silent_video, source, out_path):
    # Copies the soundtrack from the original onto the new video
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent_video, "-i", source,
                    "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "128k", "-shortest",
                    "-movflags", "+faststart", out_path], check=True)

def process_video(src, dst, edit_fn, max_side=None, report_every=25):
    # edit_fn takes a frame (float numbers, 0 to 1) and returns the edited frame
    info = video_info(src)
    w, h = target_size(info, max_side)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    silent = dst + ".silent.mp4"
    writer = VideoWriter(silent, w, h, info["fps"])
    t0, n = time.time(), 0
    try:
        for frame in read_frames(src, max_side):
            writer.write(edit_fn(frame.astype(np.float32) / 255.0))
            n += 1
            if n % report_every == 0:
                print(f"{n}/{info['frames']} frames, {time.time() - t0:.0f} s")
    finally:
        writer.close()
    add_audio(silent, src, dst)
    os.remove(silent)
    print(f"Done: {n} frames in {time.time() - t0:.0f} s -> {dst}")
    return n

def sample_frames(path, count=5, max_side=360):
    # A few evenly spaced frames, small, for quick looks
    info = video_info(path)
    wanted = sorted(set(np.linspace(0, max(info["frames"] - 1, 0), count).astype(int).tolist()))
    out, gen = [], read_frames(path, max_side)
    for i, f in enumerate(gen):
        if i in wanted:
            out.append(f.astype(np.float32) / 255.0)
        if i >= wanted[-1]:
            break
    gen.close()
    return out
