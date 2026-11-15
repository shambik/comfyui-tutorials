# Character swap in ComfyUI: Viggle Remake and akatz Character Swap V1.5

Everything you need to follow the video tutorial: the **Viggle Remake Long** node pack, Viggle's own nodes, one
ComfyUI workflow per lesson (Viggle and akatz), the source clips, and download links to every model file. Everything was tested on an **RTX 3080 10 GB** with 64 GB
of system RAM.

| Folder | What's in it |
| --- | --- |
| [`ComfyUI-Viggle-Remake-Long/`](ComfyUI-Viggle-Remake-Long/) | Our node pack: Viggle Long Swap, Viggle Character, Viggle Pick Frame, Viggle Align Keyframe, plus `install.bat` |
| [`ComfyUI-Viggle-Remake/`](ComfyUI-Viggle-Remake/) | Viggle's own Viggle Remake nodes (needed), unchanged from Viggle, Apache-2.0 |
| [`workflows/`](workflows/) | One Viggle workflow per lesson; drag a file onto the ComfyUI canvas |
| [`workflows/akatz/`](workflows/akatz/) | The akatz workflows from the video, ready to run (H3 Relay's workflow with our settings, GPL-3.0) |
| [`inputs/`](inputs/) | The AI-generated source clips used in the video (24 fps); copy them into `ComfyUI/input` |

The character pictures and edited keyframes from the video are **not** included: load your own character
pictures in the Load Image nodes, and make your own edited keyframes (see "Painted keyframes" below).

## Setup used in the video

- RTX 3080 10 GB, 64 GB RAM, Windows 10, NVIDIA driver 610.88
- ComfyUI **0.44** (MiniMax H3 support), PyTorch 2.11, CUDA 13.0, Python 3.12
- Launch flags: `python main.py --reserve-vram 0.9 --enable-dynamic-vram --async-offload 2`
- Close other GPU programs before a run: H3 sits near 9.5 GB.

## Install the nodes

1. Download this repository (Code → Download ZIP) and unzip it.
2. Double-click `01-character-swap-viggle-akatz/ComfyUI-Viggle-Remake-Long/install.bat`. It finds your ComfyUI, or
   asks you to drag the ComfyUI folder into its window. It then copies our nodes and Viggle's own nodes into
   `custom_nodes`. It does not restart anything.
3. Restart ComfyUI, double-click the canvas and type `Viggle`.

To install by hand instead (any system), copy both `ComfyUI-Viggle-Remake-Long` and `ComfyUI-Viggle-Remake` into
`ComfyUI/custom_nodes`. No extra Python packages are needed.

For akatz, install **H3 Relay**: [github.com/akatz-ai/h3-relay](https://github.com/akatz-ai/h3-relay) (GPL-3.0).

## Model files

Download each file into the ComfyUI folder shown. The models have their own licenses: read them on each page.

### Viggle Remake

| File | Size | Put it in | Download |
| --- | --- | --- | --- |
| `viggle_remake_swap_pruned_int8_convrot.safetensors` | 21 GB | `models/diffusion_models` | [Viggle/Viggle-Remake → int8/](https://huggingface.co/Viggle/Viggle-Remake/blob/main/int8/viggle_remake_swap_pruned_int8_convrot.safetensors) |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15.7 GB | `models/text_encoders` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) |
| `minimax_h3_video_vae_fp16.safetensors` | 5.2 GB | `models/vae` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_fp16.safetensors) |
| `minimax_h3_audio_vae_fp32.safetensors` | 0.6 GB | `models/vae` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors) |

### akatz Character Swap V1.5 (plus the text encoder and VAEs above)

| File | Size | Put it in | Download |
| --- | --- | --- | --- |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 21 GB | `models/diffusion_models` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) |
| `h3_character_swap_v1.5.safetensors` | 0.16 GB | `models/loras/character_swap` | [akatz-ai/MiniMax-H3-Character-Swap-LoRA](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA/blob/main/h3_character_swap_v1.5.safetensors) |
| `minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | 2 GB | `models/loras/character_swap` | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors) |
| `sam3.1_multiplex_fp16.safetensors` | 1.75 GB | `models/checkpoints` | [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1/blob/main/checkpoints/sam3.1_multiplex_fp16.safetensors) |

## The workflows

Copy `inputs/*.mp4` into `ComfyUI/input`, drag a workflow onto the canvas, and pick your own pictures in the Load
Image nodes. Times are measured on the RTX 3080 and include model loading.

| Workflow | Lesson | Clip | Time |
| --- | --- | --- | --- |
| `00_last_frame_for_joining_clips` | Take a clip's exact last frame (Pick Frame, frame -1), to start the next generated clip on it | `tut_promenade_clip1.mp4` | ~1 s |
| `01_picture_mode` | One person from a plain picture | `tut_promenade_24fps.mp4` (9.6 s) | 16.7 min |
| `02_painted_keyframe` | One person from an edited keyframe | promenade | 16.9 min |
| `03_two_people_painted` | Two people, both edited into one keyframe | `tut_cafe_24fps.mp4` | 17.0 min |
| `04_two_people_pictures` | Two people from plain pictures | cafe | 17.0 min |
| `05_people_who_come_and_go` | Frame ranges per person, two keyframes | `tut_crossing_24fps.mp4` | 17.4 min |
| `06_long_clip_25s` | The same workflow on a 25 s clip (6 windows) | `tut_promenade25_24fps.mp4` | 47.6 min |
| `07_selfie_render1_pictures_only` | Busy selfie, render 1: three characters with plain pictures | `tut_selfie15_24fps.mp4` (15.4 s) | 33 min |
| `08_selfie_render2_only_the_pilot` | Busy selfie, render 2: only the person render 1 missed | render 1's output (copy it into `ComfyUI/input` and pick it in Load Video) | 31 min |

Viggle outputs 1280×736 (0.94 MP). The source clips are 1664×928 (1.5 MP, MiniMax H3) or 1888×1056 (2 MP, LTX).

### Viggle Long Swap and the ~5 s windows

Viggle Remake takes up to 345 frames (14.4 s) per pass. On a 10 GB card about 124 frames (5.2 s) fit: 124 frames
use 9.5 GB, 141 frames 9.9 GB, and 175 frames stream weights from RAM at ~7.6 min per step. So the node's
window length defaults to 124 frames (5.17 s). Viggle Long Swap splits longer clips into windows. Each window starts on the
previous window's last 17 frames, and its last frame is pinned as the next window's keyframe, so the joins don't
show. On a card with more VRAM, raise **window_seconds** (or window_frames, they're linked; right under the model files)
up to 14.38 s to get fewer joins (not tested here). Every frame number has
a linked **seconds** field: type the time you see in the player and the frame number follows.

Plan for about 7 minutes per 5 seconds of video. On 64 GB of RAM, keep one run to about 25 s.

### Painted keyframes (edited frames)

A painted keyframe is one frame of your video, edited in an image editor or image model so the new person stands
where the old one stood, in the same pose.

1. **Viggle Pick Frame** at the moment you want (first ~4.5 s), with `editor_margin` 7 if your image model zooms in
   when it edits. Connect **Save Image**, select it, and press the **blue play button** above it ("Execute to
   selected output nodes"). Only that branch runs, and the frame is saved in about a second.
2. Edit it.
3. **Viggle Align Keyframe** (the video, your edited image, the same time) lines it up with the source frame and
   matches the colours. Connect it to `keyframe_1`, with the same time in `keyframe_1_seconds`.

### People who walk in later, and talking faces (workflows 07 and 08)

An edited keyframe makes the new face copy its expression, which holds a talking mouth still. A later window only
sees what the window before it showed. So the busy selfie uses two renders:

1. **Render 1, plain pictures only** (workflow 07): every character gets a picture, with no keyframe. The faces
   follow the talking. Viggle may miss someone who only walks in later.
2. **Render 2, only the missed person** (workflow 08): render 1 is the source video, and one Viggle Character
   remains, the missed person with their picture. `keyframe_1` is render 1's own frame (unedited). For each moment
   they walk in: Pick Frame on render 1, edit **only them** in, then **Viggle Align Keyframe** with
   **keep_unchanged** on. Every other pixel stays exactly as rendered, so nobody else can change.

Never give a later window a frame that redraws people who are already in the shot: they jump to that version.

## akatz Character Swap V1.5

The workflows in [`workflows/akatz/`](workflows/akatz/) are H3 Relay's `H3-Character-Swap.json` with the settings
from the video already set, and Create Video's `chroma_subsampling` already fixed. They are GPL-3.0, like H3 Relay.

| Workflow | Lesson | Clip |
| --- | --- | --- |
| `a1_one_person` | One person from a picture | promenade |
| `a2_talking_closeup` | Talking close-up | `tut_closeup_24fps.mp4` |
| `a3_crossing_limit` | The limit: the mask jumps to the man walking in | crossing |
| `a3c_crossing_trimmed_fix` | The fix: duration trimmed to 3.6 s | crossing |
| `a4_two_people_pass1_left` | Two people, pass 1: the man on the left | cafe |
| `a4b_two_people_pass2_right` | Pass 2: the man on the right; the source is pass 1's output (copy it into `ComfyUI/input` and pick it) | pass 1's output |

Settings used in the video:

- **Prepare Source:** resolution `480` on 10 GB (outputs 960×512); `duration_seconds` = your clip length.
- **SAM description:** the person to replace, by what you can see. **Select Mask:** `largest` / `leftmost` / `rightmost`.
- **LoRAs:** Turbo 8-step + Character Swap V1.5. **Sampling memory:** 6 GB.
- **On ComfyUI 0.44,** set Create Video's `chroma_subsampling` to `auto` before the first run. It loads as `none`,
  and ComfyUI then silently skips Save Video.
- One person per pass. For two people, run again with the first result as the source.
- If the person leaves the shot, the mask can jump to someone else. Trim the clip to the part with your person
  (`duration_seconds`).

Measured: 9.6 s, one person: 12.1 min · 10 s close-up: 13 min · 3.6 s trimmed: 4.2 min · two people (two passes): 24 min.

## Credits

[Viggle Remake](https://huggingface.co/Viggle/Viggle-Remake), [MiniMax H3 (Comfy-Org)](https://huggingface.co/Comfy-Org/MiniMax-H3),
[akatz Character Swap LoRA](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA),
[H3 Relay](https://github.com/akatz-ai/h3-relay), [lightx2v Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo),
[SAM 3.1 (Comfy-Org)](https://huggingface.co/Comfy-Org/sam3.1). The source clips were generated with MiniMax H3 and
LTX.
