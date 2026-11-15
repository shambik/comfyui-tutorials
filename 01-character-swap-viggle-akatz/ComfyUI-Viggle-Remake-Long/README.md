# Viggle Remake Long

Swap people in clips longer than one Viggle Remake pass, from one node, with no scripts and no seams.

Viggle Remake renders up to 345 frames (14.4 s at 24 fps) per pass; on a 10 GB card about 124 frames (5.2 s) fit,
which is the node's default. Set it with **window_frames** / **window_seconds** (linked, right under the model files). **Viggle Long Swap** splits a longer clip into windows of that
length and chains them:

* each window starts 17 frames before the previous one ends and takes those frames from the previous result;
* each window after the first is pinned to the previous window's last carried-over frame, so the new people keep
  their exact place and details across the join;
* it writes Viggle's prompt for every window and joins the windows with the source soundtrack.

At run time the node expands into ordinary ComfyUI nodes (the graph the H3 Mobile Bridge app runs), so ComfyUI
loads, caches and frees the models as usual.

## Nodes

| Node | What it does |
| --- | --- |
| **Viggle Character** | Who replaces whom: a name, what they look like, which person in the source they replace, an optional frame range, an optional picture. Chain up to four through `more`. |
| **Viggle Long Swap** | Source video + characters (+ optional painted keyframes) → the swapped video, its frames, and a text plan of the windows. |
| **Viggle Pick Frame** | One exact frame of a video. `-1` = the last frame. Use it to grab the frame you will paint, or the last frame of a clip to start the next clip from it. |
| **Viggle Align Keyframe** | Lines a painted frame up with its source frame (scale, shift, colour). Image models move and recolour the picture a little; Viggle snaps to a pinned frame, so that shows as a jump. |

## Install

**Windows, one double-click:** run `install.bat` from this folder. It finds your ComfyUI (portable, desktop app
or a manual install; otherwise it asks you to drag the ComfyUI folder into its window), copies the nodes into
`custom_nodes` (replacing an older copy), and checks that Viggle's own Viggle Remake nodes are installed. It does
not restart ComfyUI: restart it yourself afterwards. You can also pass the folder: `install.bat D:\ComfyUI`.

**By hand (any system):** copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI. No extra Python
packages are needed. The nodes appear under **Viggle Remake → Long**; double-click the canvas and type "Viggle".

## Step by step: picture mode (one person)

1. Double-click the canvas → **Load Video** → pick your clip (24 fps).
2. Add **Viggle Long Swap** and connect `video`.
3. Add **Save Video** and connect Viggle Long Swap's `video` output.
4. Add **Viggle Character**: type a `name`, the new person's `look` (face, hair, glasses, outfit) and in
   `replaces` the person in the source ("the man with curly grey hair and a denim jacket").
5. Add **Load Image** with a picture of the new person and connect it to `picture`.
6. Connect the character's `characters` output to Viggle Long Swap's `characters`.
7. Optional: **Preview as Text** on the `plan` output shows how the clip is split into windows.
8. Pick a seed and press Run.

## Step by step: painted keyframe

Start from the picture-mode graph and delete the Load Image.

1. Add **Viggle Pick Frame**, connect the video, set `frame` to a frame in the first ~4.5 s where the person is
   clearly visible (24 fps: 2.5 s = frame 60). Set `editor_margin` to 7 if your image model zooms in when it
   edits. Add **Save Image** to its `image` output and run once to save the frame.
2. Repaint that image with the new person in exactly the same pose.
3. Add **Load Image** with your painting, and **Viggle Align Keyframe** with the video, the painting and the same
   frame number.
4. Connect Align's `keyframe` to Viggle Long Swap's `keyframe_1` and set `keyframe_1_frame` to the same number.

Two people: add a second Viggle Character and connect its `characters` output to the first one's `more` input
(up to four). Paint everyone into the keyframe, or connect a picture to every character.

People who come and go: set each character's `first_frame` / `last_frame`, and paint a second keyframe
(`keyframe_2`) for the person who appears later, still in the first window.

Every frame number has a **seconds** field next to it: type the time you see in the video player and the
frame number follows (and the other way round). 24 fps.

People who walk in or come back, and keeping mouths free (tested on a 15 s talking selfie with three people, 2026-11-15):

1. **Render 1, pictures only:** a plain picture on every Viggle Character, no edited keyframe. Nothing pins a frame, so
   the new faces follow the talking. Viggle may miss someone who only walks in later.
2. **Render 2, only the missed person:** render 1 is the source video, one Viggle Character (the missed person, with
   their picture). `keyframe_1` = render 1's own frame from **Viggle Pick Frame** (unedited), so the first window stays
   as it is. For each moment they walk in: Pick Frame on render 1, edit **only them** into it, **Viggle Align Keyframe**
   with **keep_unchanged** on (every other pixel stays exactly as render 1 has it), into the next keyframe input.
3. A note on Viggle Long Swap: everyone else stays exactly as they are.

Never give a later window a frame that redraws people who are already in the shot: they jump to that version. And where
an edited keyframe sits, the new face copies its expression, which holds a talking mouth still.

Up to four keyframes: `keyframe_1` to `keyframe_4` (Viggle takes up to four pictures per window), each with its
own time; at least one in the first window. The order doesn't matter; the node sorts them by time. The order of the inputs doesn't matter; the node sorts them by frame.

## Two ways to swap

**Picture mode** (simplest): connect a plain picture of the new person to each Viggle Character. Nothing else.

**Painted mode** (most faithful): paint one source frame with the new people in the same pose.

1. **Viggle Pick Frame** → Save Image: pick a frame in the first ~4.5 s where every person is clearly visible.
2. Repaint it in any image editor or image model.
3. **Viggle Align Keyframe** (source video, same frame number, your painting) → `keyframe_1` of Viggle Long Swap,
   and the same frame number in `keyframe_1_frame`.

Paint only for the first window. The node refuses keyframes later in the clip, because the later windows get
their keyframe from the previous window automatically.

If someone walks in later in the first window, paint a second frame for them (`keyframe_2`) and give each
Viggle Character its frame range (`first_frame` / `last_frame`), so windows they are not in leave them out.

## Joining clips you generated separately

To make clip 2 continue clip 1 without a jump, start clip 2 from clip 1's **exact** last frame:
**Load Video** (clip 1) → **Viggle Pick Frame** (`frame` = -1) → the first-frame input of your video model.
When you join them, drop clip 2's first frame (it repeats clip 1's last one).

## Long clips

The node plans as many windows as the clip needs; nothing else changes. Measured on the RTX 3080 10 GB with 64 GB
of system RAM: a 25 s clip (600 frames, 6 windows, painted keyframe) took 47.6 min, with no visible joins. Each
window decodes only its own frames, but the swapped frames are kept in RAM until the video is saved, and
ComfyUI's cache fills free RAM: at the end of that run free RAM fell to 0.7 GB for about 20 s. On 64 GB, keep one
run to about 25 s.

In painted mode everyone must be visible in the first window (they come from your painted keyframes); the node
refuses otherwise. Use picture mode for people who appear later.

## Requirements

* ComfyUI 0.44 or newer (MiniMax H3 support).
* The **Viggle Remake** nodes (`RemakeConditioning`, `RemakeNoise`, `RemakePrefix`, `RemakeKeyframeUntil`).
* Models: the Viggle Remake swap model (`viggle_remake_swap_pruned_int8_convrot.safetensors`), the H3 text
  encoder and the H3 video and audio VAEs. Pick other file names in the node if yours differ.
* The source video at **24 fps** (`ffmpeg -i in.mp4 -r 24 out_24fps.mp4`), at least 3.75 s long.

## Tested

The graph this node builds was tested on an RTX 3080 10 GB with 64 GB RAM: 9.6 s clips (two windows) in picture
mode and painted mode, one and two people, and people who walk in and out. That took about 16.5 minutes per
clip, model loading included. The output is 1280×736 (16:9). `tests/test_expand.py` checks that the node builds
exactly those graphs. Through the node itself: five 9.6 s swaps (picture, painted, two people, two pictures, people
who come and go) took 1003-1043 s each, and a 25 s painted swap 2856 s, with no visible jump at any pin or join.

ComfyUI flags used: `--reserve-vram 0.9 --enable-dynamic-vram --async-offload 2`.
