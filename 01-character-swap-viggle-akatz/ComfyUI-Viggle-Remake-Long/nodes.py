# SPDX-License-Identifier: Apache-2.0
"""Viggle Remake Long: swap people in clips longer than one Viggle pass, from a single node.

Viggle Remake renders up to 124 frames (~5.2 s at 24 fps) per pass. ViggleLongSwap splits a longer clip into
windows, chains every window to the previous one (17 carried-over frames, plus the previous window's last
carried-over frame as its keyframe), writes Viggle's prompt for each window and joins the results with the source
soundtrack. At run time the node expands into the same graph the H3 Mobile Bridge app runs (stock nodes plus the
Viggle Remake Conditioning / Noise / Prefix / Keyframe Until nodes), so ComfyUI caches and frees memory as usual.

Viggle Character   who replaces whom (name, look, the person in the source, optional frame range, optional picture)
Viggle Long Swap   the source video + characters + optional painted keyframes -> the swapped video
Viggle Pick Frame  one frame of a video (-1 = the last one): the frame to paint, or the opening frame of the next clip
Viggle Align Keyframe   lines a painted frame up with its source frame (framing + colour) so the pin does not jump
"""
import logging

import av
import numpy as np
import torch

import folder_paths
from comfy_execution.graph_utils import GraphBuilder

from .align import align_keyframe

FPS = 24
MAX_FRAMES, MIN_FRAMES, LOCK = 124, 90, 17
SHORT_EDGE = 736
SIGMAS = "1.0, 0.990476, 0.965243, 0.923077, 0.8, 0.631579, 0.0"
PREV = "<previous window>"
DEFAULTS = {"diffusion_models": "viggle_remake_swap_pruned_int8_convrot.safetensors",
            "text_encoders": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "video_vae": "minimax_h3_video_vae_fp16.safetensors",
            "audio_vae": "minimax_h3_audio_vae_fp32.safetensors"}


# ---------------------------------------------------------------- planning (same rules as the app)

def ts(frame: int) -> str:
    s = frame / FPS
    return f"{int(s // 60):02d}:{s % 60:06.3f}"


def window_frames(available: int, longest: int = MAX_FRAMES) -> int:
    n = (min(available, longest) - 5) // 17
    frames = 17 * n + 5
    return frames if frames >= MIN_FRAMES else 0


def plan_windows(total: int, longest: int = MAX_FRAMES) -> list[tuple[int, int]]:
    """(first source frame, frame count) per window: windows of `longest` frames (17n+5; 124 = ~5.2 s, what fits a
    10 GB card) overlapping by 17, the last one sized (17n+5, at least 90) so it ends exactly on the clip's last
    frame. With longest=124 the last window is 90, 107 or 124 frames, as the app does."""
    if total <= longest:
        frames = window_frames(total, longest)
        return [(0, frames)] if frames else []
    windows, end = [(0, longest)], longest
    while end < total:
        need = total - (end - LOCK)
        if need <= longest:
            frames = next(f for f in range(MIN_FRAMES, longest + 1, 17) if f >= need)
            windows.append((total - frames, frames))
            end = total
        else:
            windows.append((end - LOCK, longest))
            end += longest - LOCK
    return windows


def keep_ranges(windows):
    """Which of its own frames each window contributes to the joined video."""
    out = []
    for k, (start, frames) in enumerate(windows):
        first = 0 if k == 0 else LOCK
        last = windows[k + 1][0] + LOCK - start if k + 1 < len(windows) else frames
        out.append((first, last))
    return out


def in_window(chars: list[dict], start: int, frames: int) -> list[int]:
    out = [i for i, c in enumerate(chars) if c["first"] < start + frames and (c["last"] < 0 or c["last"] >= start)]
    if not out:
        raise ValueError(f"Viggle Long Swap: no character is set for source frames {start}-{start + frames - 1}. "
                         "Check the first/last frame of your Viggle Character nodes.")
    return out


def prompt(chars: list[dict], idx: list[int], refs: list, total: int, notes: str) -> str:
    """Viggle's six-section prompt for one window; the source is described by the whole clip's length (as tested)."""
    chars = [chars[i] for i in idx]
    painted = any(pin is not None for _, pin in refs)
    mixed = any(ref == PREV for ref, _ in refs) and any(pin is None for _, pin in refs)
    label = f"a {total / FPS:.1f} s single continuous shot with no cut"
    notes = (" " + notes.strip()) if notes and notes.strip() else ""
    if mixed:
        # the previous window's last frame (pinned) + the plain picture of each person who walks back in
        subj = {ci: j for j, ci in enumerate(idx, 1)}
        pictures, retention = [], []
        for i, (ref, pin) in enumerate(refs, 1):
            if ref == PREV:
                pictures.append(f"<Picture {i}> is a frame of the target video from [Shot 1], at the start of this part.")
                retention.append(f"<Picture {i}>: reference - a frame of the target video. Everyone who is in it keeps "
                                 f"exactly the face, hair and outfit they have there.")
            else:
                pictures.append(f"<Picture {i}> is a character picture of <Subject {subj[ref]}>, not a frame of the "
                                f"target video.")
                retention.append(f"<Picture {i}>: reference - only <Subject {subj[ref]}>'s face, hair and outfit are "
                                 f"taken from <Picture {i}>; its background, layout and pose are not used.")
    elif painted:
        pictures = [f"<Picture {i}> is a frame of the target video from [Shot 1]." for i in range(1, len(refs) + 1)]
        retention = [f"<Picture {i}>: reference - a frame of the target video at {ts(pin)}. Every new person as they "
                     f"appear there is taken from <Picture {i}>." for i, (_, pin) in enumerate(refs, 1)]
    else:
        pictures = ["Each <Picture N> is a character picture, not a frame of the target video: " +
                    ", ".join(f"<Picture {i}> shows <Subject {i}>" for i in range(1, len(chars) + 1)) + "."]
        retention = [f"<Picture {i}>: reference - only <Subject {i}>'s face, hair and outfit are taken from "
                     f"<Picture {i}>; its background, layout and pose are not used." for i in range(1, len(chars) + 1)]
    subjects = [f"<Subject {i}> is {c['name']}: {c['look']}. {c['name']} replaces {c['replaces']} in <Video 1>."
                for i, c in enumerate(chars, 1)]
    keep = [f"<Subject {i}>: fully_preserved - {c['name']}'s face, hair and outfit stay identical in every frame."
            for i, c in enumerate(chars, 1)]
    who = "; ".join(f"<Subject {i}> in place of {c['replaces']}" for i, c in enumerate(chars, 1))
    match = (", matching " + ", ".join(f"<Picture {i}> at {ts(pin)}" for i, (_, pin) in enumerate(refs, 1)
                                       if pin is not None)) if painted and not mixed else ""
    n = len(chars)
    nl = "\n"
    return f"""subject_definitions:
<Video 1> is the source video for the editing task: {label}.{notes}
{nl.join(pictures)}
{nl.join(subjects)}

summary:
[video editing + character replacement] The target video is an edited version of <Video 1> in which {n} {'person is' if n == 1 else 'people are'} replaced: {', '.join(f'<Subject {i}>' for i in range(1, n + 1))}, each looking exactly as in the pictures and doing everything the person they replace does. Everyone else, the place and the light are unchanged. The shots, the camera, the motion and the timing of <Video 1> are kept.

retention_analysis:
<Video 1> (source video editing): fully_preserved - the shot structure, every camera angle and move, the place, the light, every other person, and the motion and timing of every action and word are copied frame for frame.
{nl.join(retention)}
{nl.join(keep)}

detailed_description:
The target video is a realistic live-action video with the same place, light and look as <Video 1>.{notes}
[Shot 1] At 00:00.000 the shot of <Video 1> with {who}; everyone moves, acts and speaks exactly as in <Video 1>{match}.
Each new person keeps their own face, hair and outfit in every frame, and none of the replaced people's faces, hair or clothes remain. Only these people change, and everything else stays as in <Video 1>.

overall_soundscape:
The source soundtrack continues throughout, unchanged.

non_diegetic_music:
N/A
"""


def plan(total: int, chars: list[dict], keyframes: list[tuple[str, int]], longest: int = MAX_FRAMES):
    """keyframes: (input name, source frame). Returns windows, keeps, refs per window ((name or PREV, pin) or
    (character index, None) in picture mode), characters per window."""
    windows = plan_windows(total, longest)
    if not windows:
        raise ValueError(f"Viggle Long Swap: the clip has {total} frames; Viggle needs at least {MIN_FRAMES} "
                         f"({MIN_FRAMES / FPS:.2f} s at 24 fps).")
    keeps = keep_ranges(windows)
    who = [in_window(chars, s, f) for s, f in windows]
    if keyframes:
        def owner(frame):
            for k, ((start, _), (a, b)) in enumerate(zip(windows, keeps)):
                if start + a <= frame < start + b:
                    return k
            return None

        owned = {}
        for name, frame in keyframes:
            k = owner(frame)
            if k is None:
                raise ValueError(f"Viggle Long Swap: {name} is at frame {frame}, outside the clip (0-{total - 1}).")
            owned.setdefault(k, []).append((name, frame - windows[k][0]))
        if 0 not in owned:
            raise ValueError("Viggle Long Swap: put at least one edited keyframe in the first window "
                             f"(frames 0-{keeps[0][1] - 1}, {ts(0)}-{ts(keeps[0][1] - 1)}).")
        per = [owned[0]]
        for k in range(1, len(windows)):
            pics = [(ci, None) for ci in who[k] if chars[ci]["picture"] is not None]
            if owned.get(k):     # a frame of the rendered video with only the newcomer edited in (see Align Keyframe)
                per.append([(PREV, LOCK - 1)] + owned[k])
            elif pics:           # Viggle's rule: pins for all of a window's pictures or none
                per.append([(PREV, None)] + pics)
            else:
                per.append([(PREV, LOCK - 1)])
        for c in chars:
            k = owner(c["first"]) if c["first"] < total else None
            if k and not owned.get(k) and c["picture"] is None:
                raise ValueError(
                    f"Viggle Long Swap: {c['name']} first appears in window {k + 1}, and nothing there shows them. Add a "
                    "keyframe in that window (a frame of your first render with only them edited in), or connect a "
                    "picture of them to their Viggle Character.")
    else:
        missing = [chars[i]["name"] for i in range(len(chars)) if chars[i]["picture"] is None]
        if missing:
            raise ValueError("Viggle Long Swap: connect a picture to every Viggle Character (missing: " +
                             ", ".join(missing) + "), or connect a painted keyframe.")
        per = [[(i, None) for i in w] for w in who]
    if any(len(r) > 4 for r in per):
        raise ValueError("Viggle Long Swap: Viggle takes up to 4 pictures per window.")
    return windows, keeps, per, who


def describe(windows, keeps, per, chars) -> str:
    lines = []
    for k, ((start, frames), (a, b), refs) in enumerate(zip(windows, keeps, per), 1):
        pics = []
        for ref, pin in refs:
            if ref == PREV:
                pics.append(f"window {k - 1}'s last carried-over frame" + (f", pinned at {pin}" if pin is not None else ""))
            elif pin is not None:
                pics.append(f"{ref} (frame {start + pin}) pinned at {pin}" if start else f"{ref} pinned at {pin}")
            elif isinstance(ref, int):
                pics.append(f"{chars[ref]['name']}'s picture")
            else:
                pics.append(f"{ref} as a look reference (not pinned)")
        lines.append(f"Window {k}: source frames {start}-{start + frames - 1} ({ts(start)}-{ts(start + frames)}), "
                     f"keeps {start + a}-{start + b - 1}; " + ", ".join(pics))
    return "\n".join(lines)


# ---------------------------------------------------------------- helpers

def _names(kind: str):
    names = folder_paths.get_filename_list(kind)
    return names or [""]


def _default(names, wanted):
    for n in names:
        if n.replace("\\", "/").split("/")[-1] == wanted:
            return n
    return names[0]


def _link(dynprompt, node_id, name):
    value = dynprompt.get_node(node_id)["inputs"].get(name)
    return value if isinstance(value, list) and len(value) == 2 else None


def read_frame(video, index: int) -> torch.Tensor:
    """One frame of a VIDEO as a [1, H, W, 3] image, decoded without loading the whole clip; -1 = the last."""
    total = video.get_frame_count()
    if index < 0:
        index += total
    if not 0 <= index < total:
        raise ValueError(f"Frame {index} is outside the video (frames 0-{total - 1}).")
    start, _ = video.get_active_trim_window() if hasattr(video, "get_active_trim_window") else (0, 0)
    src = video.get_stream_source() if hasattr(video, "get_stream_source") else None
    if src is None or start:
        return video.get_components().images[index:index + 1]
    if hasattr(src, "seek"):
        src.seek(0)
    with av.open(src, mode="r") as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i == index:
                rgb = frame.to_ndarray(format="rgb24")
                return torch.from_numpy(rgb.astype(np.float32) / 255.0)[None]
    raise ValueError(f"Could not decode frame {index}.")


# ---------------------------------------------------------------- nodes

def audio_only(video):
    """The soundtrack of a VIDEO exactly as ComfyUI's Get Video Components returns it, without decoding frames."""
    src = video.get_stream_source()
    if hasattr(src, "seek"):
        src.seek(0)
    start, duration = video.get_active_trim_window() if hasattr(video, "get_active_trim_window") else (0, 0)
    if start or duration:
        return video.get_components().audio
    with av.open(src, mode="r") as container:
        streams = [s_ for s_ in container.streams.audio if s_.codec_context is not None]
        if not streams:
            return None
        stream = streams[-1]
        resampler = av.audio.resampler.AudioResampler(format="fltp")
        chunks = []
        for packet in container.demux(stream):
            for frame in packet.decode():
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray())
        if not chunks:
            return None
        data = np.concatenate(chunks, axis=1)
        return {"waveform": torch.from_numpy(data).unsqueeze(0), "sample_rate": int(stream.sample_rate or 1)}


class ViggleVideoFrames:
    """Frames start..start+length of a VIDEO, decoded on their own (a long clip is never decoded whole)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",), "start": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                             "length": ("INT", {"default": 124, "min": 1, "max": 100000})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Viggle Remake/Long/parts"

    def run(self, video, start, length):
        fps = float(video.get_frame_rate())
        part = video.as_trimmed(start / fps, length / fps, strict_duration=False) if start or length else video
        images = part.get_components().images if part is not None else None
        if images is None or images.shape[0] < length:
            logging.warning("Viggle Video Frames: trimmed decode gave %s frames, slicing the full clip instead",
                            None if images is None else images.shape[0])
            images = video.get_components().images[start:start + length]
        return (images[:length],)


class ViggleVideoAudio:
    """The soundtrack of a VIDEO without decoding its frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",)}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "Viggle Remake/Long/parts"

    def run(self, video):
        return (audio_only(video),)


class ViggleCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "name": ("STRING", {"default": "Suit man", "tooltip": "A short name for the new person."}),
            "look": ("STRING", {"multiline": True, "default": "", "tooltip":
                     "What the new person looks like: face, hair, glasses, outfit. Be specific."}),
            "replaces": ("STRING", {"default": "the man in the yellow raincoat", "tooltip":
                         "Which person in the source video to replace, described so nobody else matches: "
                         "position, hair, clothes."}),
            "first_frame": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip":
                            "First source frame this person is in (24 fps)."}),
            "first_seconds": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 100000.0, "step": 0.01, "round": 0.01,
                         "tooltip": "The same moment in seconds (24 fps). Linked to the frame number: change either one and the other follows."}),
            "last_frame": ("INT", {"default": -1, "min": -1, "max": 100000, "tooltip":
                           "Last source frame this person is in; -1 = to the end. Windows outside the range leave "
                           "this person out of the prompt."}),
            "last_seconds": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100000.0, "step": 0.01, "round": 0.01,
                         "tooltip": "The same moment in seconds (24 fps). Linked to the frame number: change either one and the other follows. -1 = to the end."}),
        }, "optional": {
            "picture": ("IMAGE", {"tooltip": "A plain picture of the new person. Picture mode: every character needs "
                                             "one. With edited keyframes: connect one for anyone who walks in or comes "
                                             "back after the first window."}),
            "more": ("VIGGLE_CHARACTERS", {"tooltip": "Chain another Viggle Character here for two to four people."}),
        }, "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIGGLE_CHARACTERS",)
    RETURN_NAMES = ("characters",)
    FUNCTION = "run"
    CATEGORY = "Viggle Remake/Long"
    DESCRIPTION = "Who replaces whom. Chain up to four of these and connect the last one to Viggle Long Swap."

    def run(self, name, look, replaces, first_frame, last_frame, picture=None, more=None, dynprompt=None,
            unique_id=None, **_):
        if not look.strip() or not replaces.strip():
            raise ValueError(f"Viggle Character '{name}': fill in both 'look' and 'replaces'.")
        me = {"name": name.strip(), "look": look.strip().rstrip("."), "replaces": replaces.strip().rstrip("."),
              "first": first_frame, "last": last_frame,
              "picture": _link(dynprompt, unique_id, "picture") if picture is not None else None}
        return ((list(more or []) + [me]),)


class ViggleLongSwap:
    @classmethod
    def INPUT_TYPES(cls):
        unets, encoders, vaes = _names("diffusion_models"), _names("text_encoders"), _names("vae")
        return {"required": {
            "video": ("VIDEO", {"tooltip": "The source clip at 24 fps (Load Video). Any length from 3.75 s."}),
            "characters": ("VIGGLE_CHARACTERS",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "model": (unets, {"default": _default(unets, DEFAULTS["diffusion_models"])}),
            "text_encoder": (encoders, {"default": _default(encoders, DEFAULTS["text_encoders"])}),
            "video_vae": (vaes, {"default": _default(vaes, DEFAULTS["video_vae"])}),
            "audio_vae": (vaes, {"default": _default(vaes, DEFAULTS["audio_vae"])}),
        }, "optional": {
            # field order on the node: models, then the window length, then the keyframes, then notes
            "window_frames": ("INT", {"default": 124, "min": 90, "max": 345, "step": 17, "tooltip":
                              "Frames per Viggle pass, at 24 fps: 124 = 5.2 s (default, what fits a 10 GB card, tested "
                              "on an RTX 3080) ... 345 = 14.4 s (Viggle's maximum). A card with more VRAM can use "
                              "longer windows and gets fewer joins (not tested here)."}),
            "window_seconds": ("FLOAT", {"default": 5.17, "min": 3.75, "max": 14.38, "step": 0.01, "round": 0.01, "tooltip":
                               "The window length in seconds, linked to window_frames: type seconds and the nearest "
                               "length Viggle allows (17n+5 frames) is set. 5.17 s fits a 10 GB card; up to 14.38 s."}),
            "keyframe_1": ("IMAGE", {"tooltip": "Optional: a frame of the source, edited in an image editor so the new "
                                                "people stand where the old ones stood (painted mode). First window only "
                                                "(before ~4.5 s); later windows continue from the previous window."}),
            "keyframe_1_frame": ("INT", {"default": 60, "min": 0, "max": 100000,
                                         "tooltip": "The source frame keyframe_1 was painted from."}),
            "keyframe_1_seconds": ("FLOAT", {"default": 2.5, "min": -1.0, "max": 100000.0, "step": 0.01, "round": 0.01,
                         "tooltip": "The same moment in seconds (24 fps). Linked to the frame number: change either one and the other follows."}),
            "keyframe_2": ("IMAGE", {"tooltip": "Optional second painted frame, e.g. for a person who walks in later "
                                                "in the first window."}),
            "keyframe_2_frame": ("INT", {"default": 110, "min": 0, "max": 100000}),
            "keyframe_2_seconds": ("FLOAT", {"default": 4.58, "min": -1.0, "max": 100000.0, "step": 0.01, "round": 0.01,
                         "tooltip": "The same moment in seconds (24 fps). Linked to the frame number: change either one and the other follows."}),
            "keyframe_3": ("IMAGE", {"tooltip": "Optional third edited frame, also in the first window."}),
            "keyframe_3_frame": ("INT", {"default": 20, "min": 0, "max": 100000}),
            "keyframe_3_seconds": ("FLOAT", {"default": 0.83, "min": -1.0, "max": 100000.0, "step": 0.01, "round": 0.01,
                         "tooltip": "The same moment in seconds (24 fps). Linked to the frame number: change either one and the other follows."}),
            "keyframe_4": ("IMAGE", {"tooltip": "Optional fourth edited frame, also in the first window."}),
            "keyframe_4_frame": ("INT", {"default": 100, "min": 0, "max": 100000}),
            "keyframe_4_seconds": ("FLOAT", {"default": 4.17, "min": -1.0, "max": 100000.0, "step": 0.01, "round": 0.01,
                         "tooltip": "The same moment in seconds (24 fps). Linked to the frame number: change either one and the other follows."}),
            "notes": ("STRING", {"multiline": True, "default": "", "tooltip":
                      "Optional extra sentence for the prompt, e.g. 'The people far down the street stay as they are.'"}),
        }, "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("video", "frames", "plan")
    FUNCTION = "run"
    CATEGORY = "Viggle Remake/Long"
    DESCRIPTION = ("Swaps people in a clip of any length: splits it into Viggle windows (window_frames: 5.2 s by default, "
                   "what fits a 10 GB card; Viggle itself takes up to 14.4 s), chains each window to the previous one so "
                   "nothing jumps at the joins, and returns the joined video with the source sound.")

    SIGMAS = SIGMAS   # Viggle's swap schedule

    def window_prompt(self, characters, who, refs, total, notes):
        return prompt(characters, who, refs, total, notes)

    def run(self, video, characters, seed, model, text_encoder, video_vae, audio_vae, keyframe_1=None,
            keyframe_1_frame=60, keyframe_2=None, keyframe_2_frame=110, notes="", window_frames=124, keyframe_3=None,
            keyframe_3_frame=20, keyframe_4=None, keyframe_4_frame=100, dynprompt=None, unique_id=None, **_):
        fps = float(video.get_frame_rate())
        if abs(fps - FPS) > 0.01:
            raise ValueError(f"Viggle Long Swap: the source is {fps:g} fps; Viggle needs 24 fps. Convert it first, "
                             "e.g. ffmpeg -i in.mp4 -r 24 out_24fps.mp4")
        total = video.get_frame_count()
        width, height = video.get_dimensions()
        long = SHORT_EDGE * 16 // 9 // 32 * 32
        cw, ch = (SHORT_EDGE, long) if height > width else (long, SHORT_EDGE)
        # up to 4 painted keyframes (Viggle's limit per window), in frame order whatever input they came in on
        keyframes = sorted(((n, f) for n, img, f in (("keyframe_1", keyframe_1, keyframe_1_frame),
                                                     ("keyframe_2", keyframe_2, keyframe_2_frame),
                                                     ("keyframe_3", keyframe_3, keyframe_3_frame),
                                                     ("keyframe_4", keyframe_4, keyframe_4_frame)) if img is not None),
                           key=lambda k: k[1])
        if len({f for _, f in keyframes}) < len(keyframes):
            raise ValueError("Viggle Long Swap: two keyframes have the same frame number; give each its own frame.")
        windows, keeps, per, who = plan(total, characters, keyframes, window_frames)
        if getattr(self, "unpin_first", False):   # window 1's keyframes as look references (Viggle Long Edit)
            per[0] = [(name, None) for name, _ in per[0]]
        text = describe(windows, keeps, per, characters)
        logging.info("Viggle Long Swap plan (%d frames):\n%s", total, text)

        g = GraphBuilder()
        unet = g.node("UNETLoader", unet_name=model, weight_dtype="default")
        attn = g.node("ModelAttentionBackend", model=unet.out(0), attention="comfy kitchen attention")
        until = g.node("RemakeKeyframeUntil", model=attn.out(0), keyframe_until=0.9)
        clip = g.node("CLIPLoader", clip_name=text_encoder, type="minimax")
        vae = g.node("VAELoader", vae_name=video_vae)
        avae = g.node("VAELoader", vae_name=audio_vae)
        # each window decodes only its own source frames, so RAM does not grow with the clip's length
        video_link = _link(dynprompt, unique_id, "video")
        sound = g.node("ViggleVideoAudio", video=video_link)
        sampler = g.node("KSamplerSelect", sampler_name="euler")
        sigmas = g.node("ManualSigmas", sigmas=self.SIGMAS)
        pictures = {}
        for refs in per:
            for ref, pin in refs:
                if ref == PREV or ref in pictures:
                    continue
                if pin is not None:   # painted keyframe: cover-crop onto Viggle's canvas exactly like the source
                    fit = g.node("ImageScale", image=_link(dynprompt, unique_id, ref), upscale_method="lanczos",
                                 width=cw, height=ch, crop="center")
                    pictures[ref] = fit.out(0)
                elif isinstance(ref, int):
                    pictures[ref] = characters[ref]["picture"]
                else:                 # an unpinned keyframe (Viggle Long Edit): a look reference, not cropped or pinned
                    pictures[ref] = _link(dynprompt, unique_id, ref)
        keep_links, prev_decode = [], None
        for k, ((start, frames), (a, b), refs) in enumerate(zip(windows, keeps, per)):
            src = g.node("ViggleVideoFrames", video=video_link, start=start, length=frames)
            audio = g.node("TrimAudioDuration", audio=sound.out(0), start_index=start / FPS, duration=frames / FPS)
            cond = {"clip": clip.out(0), "vae": vae.out(0), "audio_vae": avae.out(0), "source": src.out(0),
                    "audio": audio.out(0), "prompt": self.window_prompt(characters, who[k], refs, total, notes),
                    "frames": frames, "short_edge": SHORT_EDGE,
                    "pins": ", ".join(str(p) for _, p in refs if p is not None)}
            for i, (ref, _) in enumerate(refs, 1):
                if ref == PREV:   # the previous window's last carried-over frame, where it lands in this window
                    last = g.node("ImageFromBatch", image=prev_decode, batch_index=start - windows[k - 1][0] + LOCK - 1,
                                  length=1)
                    cond[f"ref_{i}"] = last.out(0)
                else:
                    cond[f"ref_{i}"] = pictures[ref]
            c = g.node("RemakeConditioning", **cond)
            noise = g.node("RemakeNoise", positive=c.out(0), seed=(seed + k) % 2 ** 63)
            guider = g.node("BasicGuider", model=until.out(0), conditioning=c.out(0))
            latent = c.out(1)
            if k:
                latent = g.node("RemakePrefix", vae=vae.out(0), latent=c.out(1), prefix=prev_decode,
                                prefix_frames=LOCK, prefix_at=start - windows[k - 1][0]).out(0)
            sample = g.node("SamplerCustomAdvanced", noise=noise.out(0), guider=guider.out(0), sampler=sampler.out(0),
                            sigmas=sigmas.out(0), latent_image=latent)
            decode = g.node("VAEDecode", samples=sample.out(0), vae=vae.out(0))
            keep = g.node("ImageFromBatch", image=decode.out(0), batch_index=a, length=b - a)
            keep_links.append(keep.out(0))
            prev_decode = decode.out(0)
        join = g.node("BatchImagesNode", **{f"images.image{i}": l for i, l in enumerate(keep_links)})
        out = g.node("CreateVideo", images=join.out(0), audio=sound.out(0), fps=float(FPS))
        return {"result": (out.out(0), join.out(0), text), "expand": g.finalize()}


EDIT_SIGMAS = "1.0, 0.983607, 0.96, 0.923077, 0.857143, 0.705882, 0.0"   # Viggle's edit schedule (remake_edit_api.json)
EDIT_MODEL = "viggle_remake_edit_pruned_int8_convrot.safetensors"


def edit_prompt(keep: str, setting_from: str, setting_to: str, refs: list, total: int, notes: str, camera: str = "") -> str:
    """Viggle edit-mode prompt for a background / setting change, after Viggle's own lava-field example: the people
    are kept, the setting is replaced; every picture is a frame of the target video (the edited keyframe in window 1,
    the previous window's last frame later)."""
    label = f"a {total / FPS:.1f} s single continuous shot with no cut"
    notes = (" " + notes.strip()) if notes and notes.strip() else ""
    nl = "\n"
    pics, keys = [], []
    for i, (_, pin) in enumerate(refs, 1):
        if pin is None:
            pics.append(f"<Picture {i}> is a reference picture showing <Subject 3> around <Subject 1>; it shows the look of "
                        f"the new setting, not a frame of the target video.")
            keys.append(f"<Picture {i}> (reference): the look of <Subject 3> - its buildings, materials, colours and light - "
                        f"is taken from <Picture {i}>; its framing and composition are not copied, because the camera "
                        f"follows <Video 1>.")
        else:
            pics.append(f"<Picture {i}> is a frame of the target video, exactly as the target video shows it at that moment.")
            keys.append(f"<Picture {i}> (target keyframe): fully_preserved - the target video passes through this exact frame.")
    cam = camera.strip().rstrip(".")
    cam_line = (f" The camera moves exactly as in <Video 1>: {cam}, so <Subject 3> moves past behind <Subject 1> the same way "
                f"the original setting does.") if cam else (" The camera moves exactly as in <Video 1>, so <Subject 3> moves "
                "past behind <Subject 1> the same way the original setting does.")
    match = ", ".join(f"<Picture {i}> at {ts(pin)}" for i, (_, pin) in enumerate(refs, 1) if pin is not None)
    matching = (", matching " + match) if match else ""
    return f"""subject_definitions:
<Video 1> is the source video for the target video edit: {label}.
{nl.join(pics)}
<Subject 1> is {keep.strip().rstrip(".")}.
<Subject 2> is the setting from <Video 1>: {setting_from.strip().rstrip(".")}.
<Subject 3> is {setting_to.strip().rstrip(".")}.

summary:
[video editing] The target video is an edited version of <Video 1>. <Subject 1> moves, acts and speaks exactly as in <Video 1>, with the same framing, camera motion and timing, but no longer in the setting of <Video 1>: everything around <Subject 1> is <Subject 3>.{notes}

retention_analysis:
<Video 1> (source video editing): fully_preserved - the camera motion, the framing, every movement of <Subject 1> and its timing are copied frame for frame.
{nl.join(keys)}
<Subject 1> (appears in [Shot 1]): fully_preserved - faces, hair, clothes and every movement are retained.
<Subject 2> (appears in [Shot 1]): replaced - the whole setting of <Video 1> is replaced by <Subject 3> in every frame; nothing of it remains anywhere.
<Subject 3> (appears in [Shot 1]): new - <Subject 3> fills everything around <Subject 1>, with light that matches it.

detailed_description:
The target video is a realistic live-action video with the same camera and timing as <Video 1>, set in <Subject 3>.{notes}
[Shot 1] From 00:00.000 to the end, one continuous shot. From the very first frame <Subject 3> fills the whole picture around <Subject 1> and stays consistent in every frame; <Subject 1> moves exactly as in <Video 1>{matching}.{cam_line}

overall_soundscape:
The soundtrack of <Video 1>, unchanged.

non_diegetic_music:
N/A
"""


class ViggleLongEdit(ViggleLongSwap):
    """Change the background / setting of a clip of any length with Viggle Remake's edit model: the people stay, the
    place around them changes. Same windows and chaining as Viggle Long Swap; the first window is pinned to one edited
    frame of the source that shows the new setting."""

    @classmethod
    def INPUT_TYPES(cls):
        base = ViggleLongSwap.INPUT_TYPES()
        req = {k: v for k, v in base["required"].items() if k != "characters"}
        unets = _names("diffusion_models")
        req["model"] = (unets, {"default": _default(unets, EDIT_MODEL)})
        opt = dict(base["optional"])
        ordered = {k: opt.pop(k) for k in ("window_frames", "window_seconds")}
        ordered["keep"] = ("STRING", {"multiline": True, "default": "", "tooltip":
                           "Who stays exactly as they are, e.g. 'the man with curly grey hair and a light-blue denim jacket'."})
        ordered["setting_from"] = ("STRING", {"multiline": True, "default": "", "tooltip":
                                   "The place in the source now, e.g. 'a sunny seaside promenade with a beach and the sea'."})
        ordered["setting_to"] = ("STRING", {"multiline": True, "default": "", "tooltip":
                                 "The new place, in detail, e.g. 'a snowy mountain village street on a sunny winter day'."})
        ordered["camera"] = ("STRING", {"multiline": True, "default": "", "tooltip":
                             "How the camera moves in the source, e.g. 'the camera tracks backward in front of him as he "
                             "walks'. The new background moves the same way."})
        ordered["pin_keyframe"] = ("BOOLEAN", {"default": False, "tooltip":
                                   "Off (recommended): the edited frame is only a look reference, so the new background "
                                   "moves with the camera. On: the video passes exactly through the edited frame, which can "
                                   "freeze the background."})
        notes = opt.pop("notes")
        ordered.update(opt)
        ordered["notes"] = notes
        return {"required": req, "optional": ordered, "hidden": base["hidden"]}

    DESCRIPTION = ("Changes the background of a clip of any length with Viggle Remake's edit model: the people stay, the "
                   "place around them changes. Connect one edited frame (the source frame with the new background) to "
                   "keyframe_1; the windows are chained like Viggle Long Swap.")
    SIGMAS = EDIT_SIGMAS

    def window_prompt(self, characters, who, refs, total, notes):
        keep, setting_from, setting_to, camera = self._edit
        return edit_prompt(keep, setting_from, setting_to, refs, total, notes, camera)

    def run(self, video, seed, model, text_encoder, video_vae, audio_vae, keep="", setting_from="", setting_to="",
            camera="", pin_keyframe=False, keyframe_1=None, **kw):
        if keyframe_1 is None:
            raise ValueError("Viggle Long Edit: connect an edited frame (the source frame with the new background) to keyframe_1.")
        if not keep.strip() or not setting_to.strip():
            raise ValueError("Viggle Long Edit: fill in 'keep' (who stays) and 'setting_to' (the new place).")
        self._edit = (keep, setting_from or "the original location", setting_to, camera)
        self.unpin_first = not pin_keyframe
        scene = [{"name": "the scene", "look": "", "replaces": "", "first": 0, "last": -1, "picture": None}]
        return super().run(video, scene, seed, model, text_encoder, video_vae, audio_vae, keyframe_1=keyframe_1, **kw)


class ViggleKeyframeAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO", {"tooltip": "The source clip."}),
            "frame": ("INT", {"default": 60, "min": -1, "max": 100000, "tooltip": "The frame that was painted."}),
            "seconds": ("FLOAT", {"default": 2.5, "min": -1.0, "max": 100000.0, "step": 0.01, "round": 0.01,
                         "tooltip": "The same moment in seconds (24 fps). Linked to the frame number: change either one and the other follows."}),
            "painted": ("IMAGE", {"tooltip": "The painted frame from your image editor."}),
        }, "optional": {
            "keep_unchanged": ("BOOLEAN", {"default": False, "tooltip":
                               "On: only the areas you edited are taken from the edited image; every other pixel stays "
                               "exactly the source frame. Use it on a frame of your rendered video where you only "
                               "edited in a newcomer, so the people already there can't change."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("keyframe", "report")
    FUNCTION = "run"
    CATEGORY = "Viggle Remake/Long"
    DESCRIPTION = ("Image editors shift, zoom and recolour the whole picture a little. Viggle snaps the video to a "
                   "pinned keyframe, so those differences show as a jump. This lines the painted frame up with the "
                   "source frame (scale + shift) and matches its colours; the people are not changed.")

    def run(self, video, frame, painted, keep_unchanged=False, **_):
        src = read_frame(video, frame)[0].numpy() * 255.0
        p = painted[0, ..., :3].cpu().numpy() * 255.0
        out, report = align_keyframe(src, p, keep_unchanged)
        return (torch.from_numpy(np.clip(out, 0, 255).astype(np.float32) / 255.0)[None], report)


class VigglePickFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO",),
            "frame": ("INT", {"default": -1, "min": -100000, "max": 100000, "tooltip":
                      "Frame number at 24 fps (2.5 s = frame 60). -1 = the last frame, -2 the one before."}),
            "seconds": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100000.0, "step": 0.01, "round": 0.01,
                         "tooltip": "The same moment in seconds (24 fps). Linked to the frame number: change either one and the other follows. -1 = the last frame."}),
            "editor_margin": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 15.0, "step": 0.5, "tooltip":
                              "Percent to zoom the picture OUT before you paint it. Some image models zoom in about "
                              "7 % on every edit; giving them that margin keeps the whole frame in the result. "
                              "Viggle Align Keyframe undoes it. 0 = the frame as it is."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "run"
    CATEGORY = "Viggle Remake/Long"
    DESCRIPTION = ("One exact frame of a video. Use it for the frame you paint, or with -1 for the last frame of a "
                   "clip, to start the next clip from it so nothing jumps where the two clips meet.")

    def run(self, video, frame, editor_margin, **_):
        total = video.get_frame_count()
        index = frame + total if frame < 0 else frame
        img = read_frame(video, frame)
        if editor_margin > 0:
            from .align import warp
            s = 1.0 - editor_margin / 100.0
            img = torch.from_numpy(warp(img[0].numpy(), s, 0, 0).astype(np.float32))[None]
        info = f"frame {index} of 0-{total - 1} ({index / FPS:.2f} s at 24 fps)"
        return (img, info)


NODE_CLASS_MAPPINGS = {"ViggleVideoFrames": ViggleVideoFrames, "ViggleVideoAudio": ViggleVideoAudio,
                       "ViggleCharacter": ViggleCharacter, "ViggleLongSwap": ViggleLongSwap, "ViggleLongEdit": ViggleLongEdit,
                       "ViggleKeyframeAlign": ViggleKeyframeAlign, "VigglePickFrame": VigglePickFrame}
NODE_DISPLAY_NAME_MAPPINGS = {"ViggleVideoFrames": "Viggle Video Frames (part)",
                              "ViggleVideoAudio": "Viggle Video Audio (part)", "ViggleCharacter": "Viggle Character", "ViggleLongSwap": "Viggle Long Swap", "ViggleLongEdit": "Viggle Long Edit (background)",
                              "ViggleKeyframeAlign": "Viggle Align Keyframe", "VigglePickFrame": "Viggle Pick Frame"}
