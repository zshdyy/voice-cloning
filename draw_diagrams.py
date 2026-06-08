import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ─────────────────────────────────────────────
# DIAGRAM 1  —  Anonymization Pipeline (5 Stages)
# ─────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(20, 9))
ax1.set_xlim(0, 20)
ax1.set_ylim(0, 9)
ax1.axis("off")
fig1.patch.set_facecolor("#F8F6F0")

STAGE_COLORS = {
    1: "#D6EAF8",
    2: "#FCE4C8",
    3: "#D5F5E3",
    4: "#EDE7F6",
    5: "#FDECEA",
}
BORDER_COLORS = {
    1: "#2E86C1",
    2: "#E67E22",
    3: "#1E8449",
    4: "#6C3483",
    5: "#C0392B",
}
METHOD_COLORS = ["#AED6F1", "#A9DFBF", "#D2B4DE"]

def rbox(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.18):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)

def text(ax, x, y, s, fs=8, weight="normal", color="black", ha="center", va="center", wrap=False):
    ax.text(x, y, s, fontsize=fs, fontweight=weight, color=color,
            ha=ha, va=va, wrap=wrap,
            multialignment="center")

def arrow(ax, x1, y1, x2, y2, color="#555555"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))

# Stage header helper
def stage_header(ax, x, cx, y_top, label, sublabel, fc, ec):
    rbox(ax, x, y_top - 0.55, cx - x - 0.05, 0.5, fc=ec, ec=ec, radius=0.12)
    text(ax, (x + cx - 0.05) / 2, y_top - 0.3, label, fs=9, weight="bold", color="white")
    text(ax, (x + cx - 0.05) / 2, y_top - 0.62, sublabel, fs=7, color="#333333")

# ── Stage boundaries ──────────────────────────
stages = [
    (0.2,  3.8),   # S1
    (4.0,  7.4),   # S2
    (7.6, 11.4),   # S3
    (11.6, 17.2),  # S4
    (17.4, 19.8),  # S5
]

stage_labels = [
    ("Stage 1", "Input"),
    ("Stage 2", "Preprocessing"),
    ("Stage 3", "Feature Extraction"),
    ("Stage 4", "Anonymization (3 Methods)"),
    ("Stage 5", "VoicePrivacy Evaluation"),
]

TOP, BOT = 8.6, 0.3
for i, ((x, cx), (lbl, sub)) in enumerate(zip(stages, stage_labels), 1):
    rbox(ax1, x, BOT, cx - x - 0.05, TOP - BOT, fc=STAGE_COLORS[i], ec=BORDER_COLORS[i], lw=2)
    stage_header(ax1, x, cx, TOP, lbl, sub, STAGE_COLORS[i], BORDER_COLORS[i])

# ── Stage 1: Input ────────────────────────────
rbox(ax1, 0.35, 6.8, 3.2, 1.25, "#EBF5FB", "#2E86C1")
text(ax1, 1.95, 7.75, "Source Audio", fs=8, weight="bold")
text(ax1, 1.95, 7.35, "(Speaker A)", fs=7.5, color="#555")
text(ax1, 1.95, 6.98, "original speech to anonymize", fs=7, color="#666")

rbox(ax1, 0.35, 4.9, 3.2, 1.55, "#EBF5FB", "#2E86C1")
text(ax1, 1.95, 6.1, "Target Voice Pool", fs=8, weight="bold")
text(ax1, 1.95, 5.72, "N candidate embeddings", fs=7.5, color="#555")
text(ax1, 1.95, 5.38, "e.g. male_leaning /", fs=7, color="#666")
text(ax1, 1.95, 5.1,  "female_leaning pool", fs=7, color="#666")

# ── Stage 2: Preprocessing ───────────────────
boxes_s2 = [
    (4.15, 6.9, "Amplitude\nNormalization", "normalize audio amplitude"),
    (4.15, 5.6, "Sample Rate\nUnification", "ffmpeg → 16 kHz mono"),
    (4.15, 4.3, "Denoising", "light / standard / strong"),
]
for bx, by, bt, bs in boxes_s2:
    rbox(ax1, bx, by, 3.1, 0.95, "#FEF5E7", "#E67E22")
    text(ax1, bx + 1.55, by + 0.62, bt, fs=8, weight="bold")
    text(ax1, bx + 1.55, by + 0.22, bs, fs=7, color="#666")
    if by > 4.3:
        arrow(ax1, bx + 1.55, by - 0.05, bx + 1.55, by - 0.35 + 0.95)

# ── Stage 3: Feature Extraction ──────────────
rbox(ax1, 7.75, 6.35, 3.45, 1.7, "#EAFAF1", "#1E8449")
text(ax1, 9.47, 7.7,  "Content Encoder", fs=8, weight="bold")
text(ax1, 9.47, 7.35, "(PPG Extraction)", fs=7.5)
text(ax1, 9.47, 6.95, "Phonetic PosteriorGrams", fs=7, color="#444")
text(ax1, 9.47, 6.65, "PPG ∈ ℝ^{T×D}  (speaker-independent)", fs=7, color="#444")

rbox(ax1, 7.75, 4.55, 3.45, 1.45, "#EAFAF1", "#1E8449")
text(ax1, 9.47, 5.65, "Speaker Embedding", fs=8, weight="bold")
text(ax1, 9.47, 5.3,  "Extraction", fs=8, weight="bold")
text(ax1, 9.47, 4.9,  "extract from each candidate", fs=7, color="#444")
text(ax1, 9.47, 4.65, "in the target voice pool", fs=7, color="#444")

# ── Stage 4: Three Methods ────────────────────
method_data = [
    ("① freevc_baseline", "Balanced Baseline",
     "PPG + target embedding\ndirect combination\n→ standard FreeVC tone transfer\n→ generate N candidate outputs\npick first",
     "#AED6F1", "#1A5276"),
    ("② metric_clarity", "Clarity-Priority",
     "iterate N candidates\nscore = α×SpecDist + β×EnvCorr\nweighted ranking\n→ select highest-score candidate\nUtility-first: best audio quality",
     "#A9DFBF", "#145A32"),
    ("③ ppg_tone", "Content-Priority",
     "PPG content fidelity eval\ncontent feature consistency first\ntone assists adjustment\nbalance content retention\n& acoustic quality",
     "#D2B4DE", "#4A235A"),
]

mx_start = 11.75
mw = 1.75
for idx, (title, sub, body, fc, ec) in enumerate(method_data):
    mx = mx_start + idx * (mw + 0.12)
    rbox(ax1, mx, 4.0, mw, 3.9, fc=fc, ec=ec, lw=1.8)
    text(ax1, mx + mw/2, 7.55, title, fs=7.5, weight="bold", color=ec)
    text(ax1, mx + mw/2, 7.2,  sub,   fs=6.8, color="#444")
    # body lines
    for li, line in enumerate(body.split("\n")):
        text(ax1, mx + mw/2, 6.8 - li*0.38, line, fs=6.5, color="#333")

    # output label at bottom
    out_labels = ["Anon Audio A\n(baseline)", "Anon Audio B\n(clarity)", "Anon Audio C\n(ppg_tone)"]
    rbox(ax1, mx + 0.1, 4.1, mw - 0.2, 0.75, fc="#FDFEFE", ec=ec, lw=1.2)
    text(ax1, mx + mw/2, 4.5, out_labels[idx], fs=7, weight="bold", color=ec)

# ── Stage 5: Evaluation ───────────────────────
metrics = [
    ("ASV-EER ↑",     "Privacy",  "auto speaker verification\nerror rate; higher = harder\nto identify → better privacy"),
    ("ASR-WER ↓",     "Utility",  "word error rate;\nlower = content\nbetter preserved"),
    ("source_sim ↓",  "Privacy",  "cosine sim to source\nspeaker; lower =\nbetter de-identification"),
    ("content_kept ↑","Utility",  "1 − WER;\nhigher = more\ncontent retained"),
]
my_positions = [7.8, 6.4, 5.0, 3.6]
for (m, cat, desc), my in zip(metrics, my_positions):
    fc = "#FDEDEC" if cat == "Privacy" else "#EAF4FB"
    ec = "#C0392B" if cat == "Privacy" else "#1A5276"
    rbox(ax1, 17.5, my - 0.1, 2.1, 1.1, fc=fc, ec=ec)
    text(ax1, 18.55, my + 0.72, m,    fs=8,   weight="bold", color=ec)
    text(ax1, 18.55, my + 0.45, f"[{cat}]", fs=6.5, color="#666")
    for li, line in enumerate(desc.split("\n")):
        text(ax1, 18.55, my + 0.2 - li*0.22, line, fs=6.3, color="#444")

rbox(ax1, 17.5, 1.1, 2.1, 0.75, fc="#FEF9E7", ec="#B7950B")
text(ax1, 18.55, 1.55, "pipeline_report.json", fs=7, weight="bold", color="#B7950B")
text(ax1, 18.55, 1.22, "complete eval report", fs=6.5, color="#666")

# ── Inter-stage arrows ────────────────────────
arrow_ys = [5.45, 5.45, 5.45, 5.0]
for i in range(4):
    x1 = stages[i][1] - 0.05
    x2 = stages[i+1][0] + 0.05
    y  = arrow_ys[i]
    arrow(ax1, x1, y, x2, y, color=BORDER_COLORS[i+2])

# Title
text(ax1, 10.0, 8.92,
     "Speech Anonymization Pipeline  —  5-Stage Processing Flow",
     fs=11, weight="bold", color="#1C2833")

plt.tight_layout(pad=0.3)
fig1.savefig("/Users/jonathan/Workspace/voice-cloning/anonymization_pipeline.svg",
             format="svg", dpi=200, bbox_inches="tight")
print("Diagram 1 saved.")

# ─────────────────────────────────────────────
# DIAGRAM 2  —  Overall System Architecture
# ─────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(22, 11))
ax2.set_xlim(0, 22)
ax2.set_ylim(0, 11)
ax2.axis("off")
fig2.patch.set_facecolor("#F8F6F0")

def rbox2(ax, x, y, w, h, fc, ec, lw=1.6, radius=0.2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)

def text2(ax, x, y, s, fs=8.5, weight="normal", color="black", ha="center", va="center"):
    ax.text(x, y, s, fontsize=fs, fontweight=weight, color=color,
            ha=ha, va=va, multialignment="center")

def arr2(ax, x1, y1, x2, y2, color="#444444", lw=1.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))

# ── Background columns ────────────────────────
rbox2(ax2, 0.15, 0.3, 3.7,  10.1, "#EBF5FB", "#2E86C1", lw=2.5)
rbox2(ax2, 4.2,  0.3, 13.8, 10.1, "#F4ECF7", "#6C3483", lw=2.5)
rbox2(ax2, 18.3, 0.3, 3.5,  10.1, "#E9F7EF", "#1E8449", lw=2.5)

text2(ax2, 2.0,  10.6, "INPUT LAYER",   fs=11, weight="bold", color="#2E86C1")
text2(ax2, 11.1, 10.6, "PROCESSING MODULES",  fs=11, weight="bold", color="#6C3483")
text2(ax2, 20.0, 10.6, "OUTPUT LAYER",  fs=11, weight="bold", color="#1E8449")

# ── Input boxes ───────────────────────────────
input_data = [
    ("① Single Audio\n(.wav / .mp3 / .m4a)", "#D6EAF8", "#2E86C1"),
    ("② Source Audio\n+ Target Reference\nAudio", "#D6EAF8", "#2E86C1"),
    ("③ Reference Audio\n+ Audio-to-Evaluate", "#D6EAF8", "#2E86C1"),
    ("④ Source Audio\n+ Target\nVoice Pool (N)", "#D6EAF8", "#2E86C1"),
]
in_ys = [8.3, 6.2, 4.2, 1.8]
for (label, fc, ec), iy in zip(input_data, in_ys):
    rbox2(ax2, 0.35, iy, 3.3, 1.55, fc=fc, ec=ec)
    text2(ax2, 2.0, iy + 0.78, label, fs=8.5, weight="bold", color="#1A3A5C")

# ── Module boxes ──────────────────────────────
module_data = [
    # (x, y, w, h, title, subtitle, body_lines, fc, ec, arrow_color)
    (4.4, 7.6, 13.4, 2.2,
     "Module 1 — WORLD Single-Audio Voice Conversion",
     "Traditional Signal Processing  |  Vocoder-Based Parametric Decoupling",
     ["DIO → StoneMask (F0 extraction)  →  CheapTrick (spectral envelope)  →  D4C (aperiodicity)  →  Synthesize",
      "pitch_ratio × F0  +  formant_ratio × spectral warping  +  AP preserved  +  strength blending"],
     "#EDE7F6", "#6C3483"),

    (4.4, 5.2, 13.4, 2.1,
     "Module 2 — Voice Cloning (Dual-Audio)",
     "Deep Learning  |  Zero-Shot Timbre Transfer",
     ["Path A — FreeVC (Coqui TTS):  PPG ContentEncoder + Speaker Encoder  →  HiFi-GAN decoder",
      "Path B — OpenVoice:  Speaker Embedding extraction  →  ToneColorConverter  (τ ∈ [0.20, 0.40])"],
     "#E8F8F5", "#1E8449"),

    (4.4, 3.5, 13.4, 1.45,
     "Module 3 — Speaker Similarity Evaluation",
     "OpenVoice Speaker Encoder  |  Cosine Similarity",
     ["extract speaker embedding from both audios  →  cosine(a,b) = a·b / (‖a‖‖b‖)  →  score ∈ [0, 100]",
      "short audio pad mechanism (<6s auto-extend)  |  reused as source_similarity in anonymization"],
     "#FEF9E7", "#B7950B"),

    (4.4, 0.55, 13.4, 2.65,
     "Module 4 — Speech Anonymization  (Core Module)",
     "PPG Decoupling + Target Pool Selection  |  VoicePrivacy Framework",
     ["Preprocess: amplitude norm  →  16 kHz mono (ffmpeg)  →  denoising (light/std/strong)",
      "Feature:  ContentEncoder → PPG  +  Speaker Embedding per candidate",
      "Method ①  freevc_baseline: PPG+embedding direct → N candidates, pick first  (balanced baseline)",
      "Method ②  metric_clarity:  score=α×SpecDist+β×EnvCorr, top-ranked  (utility/clarity-first)",
      "Method ③  ppg_tone:        PPG content fidelity priority + tone adjust  (content-first)"],
     "#FDEDEC", "#C0392B"),
]

for mx, my, mw, mh, title, sub, lines, fc, ec in module_data:
    rbox2(ax2, mx, my, mw, mh, fc=fc, ec=ec, lw=2)
    # title bar
    rbox2(ax2, mx, my + mh - 0.52, mw, 0.52, fc=ec, ec=ec, radius=0.1)
    text2(ax2, mx + mw/2, my + mh - 0.26, title, fs=9.5, weight="bold", color="white")
    text2(ax2, mx + mw/2, my + mh - 0.62, sub, fs=7.5, color="#555555")
    for li, line in enumerate(lines):
        text2(ax2, mx + mw/2, my + mh - 0.95 - li*0.38, line, fs=7.8, color="#222222")

# ── Output boxes ──────────────────────────────
output_data = [
    ("① Converted Audio\n(pitch+formant ctrl)\n+ F0 / Mel Spectrogram\nVisualization",     "#D5F5E3", "#1E8449"),
    ("② Cloned Audio\n+ Speaker Similarity\nReport (0–100)",                                "#D5F5E3", "#1E8449"),
    ("③ Anon Audio A/B/C\n+ VoicePrivacy Report\n(ASV-EER / ASR-WER\nsource_sim / content_kept)", "#D5F5E3", "#1E8449"),
]
out_ys = [8.0, 5.7, 1.0]
for (label, fc, ec), oy in zip(output_data, out_ys):
    rbox2(ax2, 18.5, oy, 3.1, 2.0, fc=fc, ec=ec)
    text2(ax2, 20.05, oy + 1.0, label, fs=8.5, weight="bold", color="#145A32")

# ── Arrows: input → module ────────────────────
connections_in = [
    (0, 8.3+0.78, 4.4, 7.6+2.2-0.26),   # input1 → mod1
    (1, 6.2+0.78, 4.4, 5.2+2.1-0.26),   # input2 → mod2
    (2, 4.2+0.78, 4.4, 3.5+1.45-0.26),  # input3 → mod3
    (3, 1.8+0.78, 4.4, 0.55+2.65-0.26), # input4 → mod4
]
for (idx, y_in, x_mod, y_mod) in connections_in:
    arr2(ax2, 3.65, y_in, x_mod, y_mod, color="#2E86C1", lw=1.8)

# ── Arrows: module → output ───────────────────
connections_out = [
    (7.6+2.2/2,   17.8, 8.0+1.0),
    (5.2+2.1/2,   17.8, 5.7+1.0),
    (0.55+2.65/2, 17.8, 1.0+1.0),
]
for (y_src, x_start, y_dst) in connections_out:
    arr2(ax2, x_start, y_src, 18.5, y_dst, color="#1E8449", lw=1.8)

# Module 3 → output (sim score is also in output 2)
arr2(ax2, 17.8, 3.5+1.45/2, 18.5, 5.7+0.5, color="#B7950B", lw=1.4)

# ── Title ─────────────────────────────────────
text2(ax2, 11.0, 10.82,
      "Speech Privacy Protection & Voice Conversion System",
      fs=13.5, weight="bold", color="#1C2833")
text2(ax2, 11.0, 10.42,
      "Traditional Signal Processing  +  Deep Learning Hybrid Architecture",
      fs=10, color="#555555")

plt.tight_layout(pad=0.3)
fig2.savefig("/Users/jonathan/Workspace/voice-cloning/system_architecture.svg",
             format="svg", dpi=200, bbox_inches="tight")
print("Diagram 2 saved.")
