"""video.py - read a video frame by frame, write frames back, detect scene cuts, grade per-shot, and track masks."""
import os, json, time, subprocess
import numpy as np
import cv2

# ---------- FFprobe and metadata ----------
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

# ---------- Reading frames with ffmpeg ----------
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
        p.terminate()
        p.wait()

# ---------- Writing frames into MP4 ----------
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

# ---------- Step 11: Process video with fixed look ----------
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
    if os.path.exists(silent):
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

# ---------- Step 12: Scene cut / shot detection ----------
def detect_scene_cuts(path, threshold=0.35, max_side=240):
    """
    Detects cut points between shots using HSV histogram differences.
    Returns list of frame indices where each shot begins.
    """
    cuts = [0]
    prev_hist = None
    info = video_info(path)
    print(f"Detecting scene cuts across {info['frames']} frames...")
    for idx, frame in enumerate(read_frames(path, max_side)):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > threshold:
                cuts.append(idx)
        prev_hist = hist
    cuts.append(info["frames"])
    print(f"Found {len(cuts) - 1} shot(s): cut indices at {cuts}")
    return cuts

# ---------- Step 13: Per-shot grading ----------
def process_video_per_shot(src, dst, shot_brain_fn, max_side=None, report_every=25):
    """
    Grades each shot individually:
    - Finds cut points
    - Calls shot_brain_fn(first_frame_of_shot, shot_idx) to determine plan for that shot
    - Applies shot-specific grade to all frames within that shot
    """
    cuts = detect_scene_cuts(src, max_side=240)
    info = video_info(src)
    w, h = target_size(info, max_side)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    silent = dst + ".silent.mp4"
    writer = VideoWriter(silent, w, h, info["fps"])

    current_shot_idx = 0
    next_cut = cuts[1] if len(cuts) > 1 else info["frames"]
    current_edit_fn = None

    t0, n = time.time(), 0
    try:
        for idx, frame_u8 in enumerate(read_frames(src, max_side)):
            frame_f32 = frame_u8.astype(np.float32) / 255.0
            if idx >= next_cut and current_shot_idx + 1 < len(cuts) - 1:
                current_shot_idx += 1
                next_cut = cuts[current_shot_idx + 1]
                current_edit_fn = None

            if current_edit_fn is None:
                print(f"[Shot {current_shot_idx + 1}] Analyzing first frame {idx}...")
                current_edit_fn = shot_brain_fn(frame_f32, current_shot_idx)

            writer.write(current_edit_fn(frame_f32))
            n += 1
            if n % report_every == 0:
                print(f"{n}/{info['frames']} frames graded ({time.time() - t0:.0f}s)")
    finally:
        writer.close()

    add_audio(silent, src, dst)
    if os.path.exists(silent):
        os.remove(silent)
    print(f"Completed per-shot grading: {n} frames -> {dst}")
    return n

# ---------- Step 14: Mask tracking across frames (Optical Flow) ----------
def propagate_mask_flow(prev_gray, curr_gray, mask):
    """
    Propagates a 2D float32 mask from prev frame to curr frame using Farneback optical flow.
    """
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = prev_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]
    warped = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR)
    return np.clip(warped, 0, 1)
