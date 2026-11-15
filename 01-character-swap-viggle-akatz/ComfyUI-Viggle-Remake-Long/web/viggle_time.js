// Viggle Remake Long: every frame number has a seconds field next to it (24 fps). Change either one and the other
// follows, so you can type the time you see in the video player instead of working out frame numbers.
// The frame number is what the nodes use; the seconds field is kept in step with it.
import { app } from "../../scripts/app.js";

const FPS = 24;
const PAIRS = {
    VigglePickFrame: [["frame", "seconds"]],
    ViggleKeyframeAlign: [["frame", "seconds"]],
    ViggleCharacter: [["first_frame", "first_seconds"], ["last_frame", "last_seconds"]],
    ViggleLongSwap: [["window_frames", "window_seconds", "window"], ...[1, 2, 3, 4].map((k) => [`keyframe_${k}_frame`, `keyframe_${k}_seconds`])],
};
const toSeconds = (f) => (f < 0 ? -1 : Math.round((f / FPS) * 100) / 100);
const toFrame = (s) => (s < 0 ? -1 : Math.round(s * FPS));
// a Viggle window is 17n+5 frames, 90..345: snap typed seconds to the nearest allowed length
const toWindow = (s) => Math.min(345, Math.max(90, Math.round((s * FPS - 5) / 17) * 17 + 5));

function link(node) {
    const pairs = PAIRS[node.comfyClass];
    if (!pairs || !node.widgets) return;
    for (const [fName, sName, kind] of pairs) {
        const fromSeconds = kind === "window" ? toWindow : toFrame;
        const wf = node.widgets.find((w) => w.name === fName);
        const ws = node.widgets.find((w) => w.name === sName);
        if (!wf || !ws || wf.__viggleLinked) continue;
        wf.__viggleLinked = true;
        let busy = false;
        const sync = (from, to, conv) => {
            if (busy) return;
            busy = true;
            const v = conv(from.value);
            if (to.value !== v) {
                to.value = v;
                to.callback?.(v);
            }
            busy = false;
            node.setDirtyCanvas?.(true, true);
        };
        const cf = wf.callback, cs = ws.callback;
        wf.callback = function (...args) { const r = cf?.apply(this, args); sync(wf, ws, toSeconds); return r; };
        ws.callback = function (...args) {
            const r = cs?.apply(this, args);
            sync(ws, wf, fromSeconds);
            if (kind === "window") sync(wf, ws, toSeconds);   // show the snapped length
            return r;
        };
        node.__viggleSync = node.__viggleSync || [];
        node.__viggleSync.push(() => { busy = true; ws.value = toSeconds(wf.value); busy = false; });
    }
    node.__viggleSync?.forEach((f) => f());
}

app.registerExtension({
    name: "ViggleRemakeLong.LinkedTime",
    nodeCreated(node) { link(node); },
    loadedGraphNode(node) {   // a saved workflow: the frame numbers are what count, the seconds follow them
        link(node);
        node.__viggleSync?.forEach((f) => f());
    },
});
