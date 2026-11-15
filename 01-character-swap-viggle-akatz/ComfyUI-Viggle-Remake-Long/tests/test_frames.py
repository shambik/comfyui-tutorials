"""Offline check of Viggle Pick Frame (-1 = exact last frame, as FFmpeg decodes it) and Viggle Align Keyframe
(same scale/shift as kf_fix.py on a tutorial keyframe).  <ComfyUI python> tests/test_frames.py <ComfyUI dir> <video> <painted> <frame>"""
import importlib.util, subprocess, sys
from pathlib import Path
import numpy as np
comfy, video, painted, frame = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, str(comfy))
pack = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vrl", pack / "__init__.py", submodule_search_locations=[str(pack)])
vrl = importlib.util.module_from_spec(spec); sys.modules["vrl"] = vrl; spec.loader.exec_module(vrl)
from comfy_api.latest import InputImpl
from PIL import Image
import torch
v = InputImpl.VideoFromFile(video)
n = v.get_frame_count()
img, info = vrl.nodes.VigglePickFrame().run(v, -1, 0.0)
w, h = v.get_dimensions()
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", "select=eq(n\\,%d)" % (n - 1), "-frames:v", "1", "-f", "rawvideo",
                      "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
ref = np.frombuffer(raw, np.uint8).reshape(h, w, 3).astype(np.float32)
d = np.abs(img[0].numpy() * 255 - ref).mean()
print("pick -1:", info, "shape", tuple(img.shape), f"mean abs diff vs ffmpeg {d:.2f}")
pimg = torch.from_numpy(np.asarray(Image.open(painted).convert("RGB"), dtype=np.float32) / 255)[None]
out, rep = vrl.nodes.ViggleKeyframeAlign().run(v, frame, pimg)
print("align:", rep, tuple(out.shape))
