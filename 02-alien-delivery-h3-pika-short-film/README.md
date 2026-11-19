# Alien Delivery: a sci-fi comedy short made in ComfyUI with MiniMax H3 + Pika

![Alien Delivery thumbnail](thumbnail.jpg)

A 106-second live-action-style sci-fi comedy, made locally on an **RTX 3080 10 GB**. A pizza delivery guy crosses an
alien canyon city to deliver a large pepperoni to Lord Kethra, on the top floor of a mushroom tower. 17 shots of 6 s,
all generated with **MiniMax H3 + the Pika H3 Flex LoRA** in ComfyUI 0.44, with H3's own sound and dialogue. They were then
upscaled to 2560×1440 and smoothed to 48 fps.

Everything to make shots like these is here: the exact ComfyUI workflows the film was rendered with, the film's own
frames to test them, download links to every model, and the method that keeps a multi-shot film continuous.

| Folder | What's in it |
| --- | --- |
| [`workflows/`](workflows/) | The three ComfyUI workflows; drag a file onto the ComfyUI canvas. Each has a **Read me** note with its models and links |
| [`workflows/api/`](workflows/api/) | The same graphs in API format (for scripts / the `/prompt` endpoint) |
| [`inputs/`](inputs/) | The film's frames and one upscaled shot, for testing; copy them into `ComfyUI/input` |

The workflows are the real graphs the film was rendered with (exported from the finished shots' metadata). The
prompt, seed and settings are those of shots D1 and E2 of the film.

## Setup used

- RTX 3080 10 GB, 64 GB RAM, Windows 10
- ComfyUI **0.44** (MiniMax H3 support), PyTorch 2.11, CUDA 13.0, Python 3.12
- Launch flags: `python main.py --reserve-vram 0.9 --enable-dynamic-vram --async-offload 2`
- Close other GPU programs before a run: H3 sits near 9.5 GB.

## Model files

Download each file into the ComfyUI folder shown. The models have their own licenses: read them on each page.

| File | Size | Put it in | Download |
| --- | --- | --- | --- |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 21 GB | `models/diffusion_models` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) |
| `pika_h3_flex_fl2va_comfy_bf16.safetensors` | 1.96 GB | `models/loras` | [Pika-Labs/Pika-H3-Flex → comfy/loras](https://huggingface.co/Pika-Labs/Pika-H3-Flex/blob/main/comfy/loras/pika_h3_flex_fl2va_comfy_bf16.safetensors) |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15.7 GB | `models/text_encoders` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) |
| `minimax_h3_video_vae_int8_convrot.safetensors` | 3.17 GB | `models/vae` | [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/minimax_h3_video_vae_int8_convrot.safetensors) |
| `minimax_h3_audio_vae_fp32.safetensors` | 0.6 GB | `models/vae` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors) |
| `rife_v4.26.safetensors` | 22 MB | `models/frame_interpolation` | [Comfy-Org/Frame-Interpolation](https://huggingface.co/Comfy-Org/Frame-Interpolation/blob/main/frame_interpolation/rife_v4.26.safetensors) |

The video VAE is **Kijai's** int8 build (3.17 GB), the file this film used. Comfy-Org's file of the same name is a different
2.81 GB build.

**Custom nodes:** none for the opening-frame and RIFE workflows: every node is built into ComfyUI 0.44. The first + last
frame workflow uses *Patch Sage Attention KJ* from [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes), which
needs SageAttention installed in ComfyUI's Python.

## The workflows

Copy `inputs/*` into `ComfyUI/input`, drag a workflow onto the canvas and queue. Times are measured on the RTX 3080.

| Workflow | What it does | Input | Time |
| --- | --- | --- | --- |
| `AlienDelivery_H3_Pika_opening_frame_int8` | One 6 s shot (1632×928, 158 frames, 24 fps) with sound and dialogue from an opening frame. H3 FL2VA + Pika LoRA, 6 steps (`lcm` with Pika's sigmas), int8 attention (built-in *Model Attention Backend → comfy kitchen attention*) | `alien_delivery_D1_opening.png` | 8.3 min |
| `AlienDelivery_H3_Pika_first_last_frame_sage` | The same with an end frame, for shots that must land on an exact picture with a locked camera | `alien_delivery_E2_first.png` + `alien_delivery_E2_last.png` | 10.8 min |
| `AlienDelivery_RIFE_x2_48fps` | RIFE 4.26 doubles the frames of one upscaled shot (24 → 48 fps) and keeps its sound | `alien_delivery_D1_2560_24fps.mp4` | ~1 min |

int8 attention was only tested with opening frames, so the first + last frame shots used SageAttention.

## How the film was made

1. **Frames.** Every shot opens on a still from Codex image generation, cropped to 1632×928. The cast was designed
   once on a character line-up sheet and passed with every frame request.
2. **Shots.** The opening-frame workflow, 6 s each. First + last frame only where both frames share one locked camera
   and every character is in both.
3. **Upscale.** NVIDIA RTX Video Super Resolution to 2560×1440 (NVIDIA Visual Enhancer `VE_CLI.exe`, outside ComfyUI),
   on the 24 fps originals.
4. **Smoothing.** The RIFE workflow, **per shot**: across a hard cut RIFE draws a ghost frame, so smooth before joining.
5. **Edit.** The shots were joined in FFmpeg (video and audio concatenated separately, cuts on matching actions).
6. **Music.** An instrumental "retro sci-fi comedy space funk" track (112 BPM) from Google Lyria 3.5. It was mixed about
   14 LU under the dialogue, with sidechain ducking whenever someone speaks.

### What made the shots feel like one film

- **One set per scene.** Draw one master view of the scene, then cut the closer shots out of it and re-detail them, keeping
  the layout. Shots drawn separately never look like the same place.
- **Cut on the action.** Each shot ends on the pose or look that the next shot opens with.
- **First + last frame only for changes in place.** It's for an in-place change seen by a locked camera, such as the
  chameleon changing colour or a door opening. Build the end frame as the start frame with only the changed part edited;
  a fully redrawn end frame makes H3 morph the set between the two.
- **Characters already in frame.** H3 invents characters that walk into a shot; put everyone in the opening frame.
- **Give everyone in frame something to do**, or they stand frozen.
- **Props stay in one hand.** Never write "puts X down … picks X up" in one shot: the prop reappears. Bites grow back too.
- **Eyelines.** A character must look at whoever they speak to, pinned in the frame itself. In a close-up with the other
  person off screen, write "keeps looking at X off screen at the right, never to camera".
- **Short English lines with a fixed voice per character**, `(S1)…(S5)`. "Chattering in clicks" turns into gibberish.
- **No steam near faces:** when someone sniffs, H3 makes steam pour out of their nose.

### Prompt format

Every shot uses the official MiniMax H3 layout. The real prompt of shot D1 (it's in the workflow):

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] A six-second live-action medium-wide shot begins exactly from
<Picture 1>, preserving Danny, ... and Lord Kethra, ... in the open pearl doorway, ... The eye-level camera holds
its position with slight small-amplitude handheld breathing. From 0.00 to 2.40 seconds, Danny, with a warm,
slightly husky young male voice and a relaxed American accent (S1), says with a bright professional smile:
<d>[English] Large pepperoni for... Lord Kethra?</d> His mouth articulates only this line. From 2.40 to 4.00
seconds, Lord Kethra looks down at the big box with curiosity and takes it from Danny in long silver fingers.
From 4.00 to 6.00 seconds, the lord holds the box in one hand and lifts the lid ... Anyone who is not speaking
keeps their mouth closed and makes no vocal sounds. Natural real-time motion at normal speed ...

overall_soundscape: Canyon wind behind them, the heavy rustle of robes, cardboard creaking.

non_diegetic_music: N/A
```

For first + last frame, add a second alignment line: `At 6.00 seconds into the target video, <Picture 2> (from [Shot 1]) is fully referenced.`

The frames, the clip and this film are AI-generated. The characters are fictional.
