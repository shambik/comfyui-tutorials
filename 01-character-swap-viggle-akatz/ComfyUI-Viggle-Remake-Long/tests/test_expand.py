"""Offline check: the graph Viggle Long Swap expands into equals the helper's tested API graph for the same job
(the tutorial's r2 promenade, painted, and r4 crossing, two keyframes + frame ranges, and r5 cafe, two pictures).

    <ComfyUI python> tests/test_expand.py <ComfyUI dir> <jobs dir>
"""
import importlib.util, json, sys
from fractions import Fraction
from pathlib import Path

comfy, jobs = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(comfy))
pack = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vrl", pack / "__init__.py", submodule_search_locations=[str(pack)])
vrl = importlib.util.module_from_spec(spec); sys.modules["vrl"] = vrl; spec.loader.exec_module(vrl)
N = vrl.nodes


class Video:
    def __init__(s, n): s.n = n
    def get_frame_rate(s): return Fraction(24)
    def get_frame_count(s): return s.n
    def get_dimensions(s): return (1664, 928)


class Dyn:
    def __init__(s, g): s.g = g
    def get_node(s, i): return s.g[i]


def run(name):
    job = json.loads((jobs / f"{name}.json").read_text(encoding="utf-8"))
    ref = json.loads((jobs / f"{name}_viggle_api.json").read_text(encoding="utf-8"))
    g = {"vid": {"inputs": {}}, "swap": {"inputs": {"video": ["vid", 0]}}}
    chars = None
    for i, c in enumerate(job["characters"]):
        cid = f"char{i}"
        g[cid] = {"inputs": {"picture": [f"pic{i}", 0]} if job.get("pictures") else {}}
        f = c.get("frames") or [0, -1]
        chars = N.ViggleCharacter().run(c["name"], c["look"], c["replaces"], f[0], f[1],
                                        picture=object() if job.get("pictures") else None, more=chars,
                                        dynprompt=Dyn(g), unique_id=cid)[0]
    kw = {}
    for k, kf in enumerate(job.get("keyframes", []), 1):
        g["swap"]["inputs"][f"keyframe_{k}"] = [f"kf{k}", 0]
        kw[f"keyframe_{k}"] = object(); kw[f"keyframe_{k}_frame"] = kf["frame"]
    out = N.ViggleLongSwap().run(Video(job["frames"]), chars, job["seed"], "m", "t", "v", "a",
                                 notes=job.get("notes", ""), dynprompt=Dyn(g), unique_id="swap", **kw)
    exp = out["expand"]
    print(name); print(out["result"][2])
    # compare by class: the sequence of (class, scalar inputs) per window
    def sig(graph, conv):
        rows = []
        comps = {i for i, n in graph.items() if n["class_type"] == "GetVideoComponents"}
        for nid, n in graph.items():
            if n["class_type"] in ("GetVideoComponents", "ViggleVideoAudio"):
                continue
            if n["class_type"] == "ImageFromBatch" and n["inputs"]["image"][0] in comps:   # helper: slice of the full decode
                rows.append(json.dumps(["SOURCE", n["inputs"]["batch_index"], n["inputs"]["length"]]))
                continue
            if n["class_type"] == "ViggleVideoFrames":                                    # node: the window decoded alone
                rows.append(json.dumps(["SOURCE", n["inputs"]["start"], n["inputs"]["length"]]))
                continue
            ins = {k: v for k, v in n["inputs"].items() if not isinstance(v, list)}
            for k in ("unet_name", "clip_name", "vae_name", "image"):
                ins.pop(k, None)
            if n["class_type"] == "RemakeConditioning":
                ins["refs"] = sum(1 for k in n["inputs"] if k.startswith("ref_"))
            rows.append(json.dumps([n["class_type"], ins], sort_keys=True))
        return sorted(rows)
    a, b = sig(exp, None), sig({k: v for k, v in ref.items() if v["class_type"] not in ("LoadVideo", "LoadImage", "SaveVideo")}, None)
    missing = [r for r in b if r not in a]; extra = [r for r in a if r not in b]
    for r in missing: print("  only in helper:", r[:300])
    for r in extra: print("  only in node:  ", r[:300])
    print("  nodes", len(exp), "helper", len(ref), "MATCH" if not missing and not extra else "DIFF")
    return not missing and not extra


ok = all([run(n) for n in sys.argv[3:]])
sys.exit(0 if ok else 1)
