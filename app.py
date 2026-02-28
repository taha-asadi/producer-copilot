# app.py
# Producer Co-Pilot (by Taha Asadi)
# Streamlit app: upload a mixdown, run one analysis, get a premium, practical breakdown.

import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

import librosa
import pyloudnorm as pyln  # LUFS meter

# IMPORTANT: set_page_config must be the first Streamlit call
st.set_page_config(
    page_title="Producer Co-Pilot (by Taha Asadi)",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Styling (premium UI)
# ----------------------------

def inject_css() -> None:
    st.markdown(
        """
        <style>
          /* App background */
          .stApp {
            background: radial-gradient(1200px 600px at 10% 0%, rgba(141, 66, 245, 0.12), rgba(0,0,0,0)) ,
                        radial-gradient(900px 500px at 90% 10%, rgba(58, 142, 255, 0.10), rgba(0,0,0,0));
          }

          /* Header card */
          .pc-header {
            border-radius: 18px;
            padding: 22px 22px 18px 22px;
            background: linear-gradient(90deg, rgba(141, 66, 245, 0.10), rgba(58, 142, 255, 0.08));
            border: 1px solid rgba(255,255,255,0.10);
          }
          .pc-title {
            font-size: 46px;
            font-weight: 800;
            line-height: 1.05;
            margin: 0;
          }
          .pc-subtitle {
            font-size: 16px;
            opacity: 0.82;
            margin-top: 6px;
          }

          /* Metric tiles */
          .pc-tile {
            border-radius: 16px;
            padding: 16px 16px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
          }
          .pc-tile-label {
            font-size: 12px;
            opacity: 0.75;
            margin-bottom: 6px;
          }
          .pc-tile-value {
            font-size: 34px;
            font-weight: 800;
            margin: 0;
          }

          /* Chips */
          .pc-chip {
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.04);
            margin-right: 8px;
            margin-bottom: 8px;
          }
          .pc-chip-danger { border-color: rgba(255, 86, 86, 0.35); background: rgba(255, 86, 86, 0.10); }
          .pc-chip-warn   { border-color: rgba(255, 186, 71, 0.35); background: rgba(255, 186, 71, 0.10); }
          .pc-chip-good   { border-color: rgba(77, 215, 120, 0.35); background: rgba(77, 215, 120, 0.10); }

          /* Green primary button */
          div.stButton > button[kind="primary"] {
            background: linear-gradient(90deg, rgba(77, 215, 120, 0.95), rgba(44, 190, 102, 0.95));
            border: 1px solid rgba(77, 215, 120, 0.30);
            color: #07120a;
            font-weight: 800;
            border-radius: 12px;
            padding: 0.65rem 1.0rem;
          }
          div.stButton > button[kind="primary"]:hover {
            filter: brightness(1.03);
          }

          /* Reduce padding at top */
          .block-container { padding-top: 1.25rem; }

          /* Footer */
          .pc-footer {
            opacity: 0.75;
            font-size: 13px;
            margin-top: 18px;
          }
          .pc-footer a { text-decoration: none; font-weight: 700; }

        </style>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# Helpers
# ----------------------------

BAND_ORDER = [
    "Sub (20–60 Hz)",
    "Low (60–200 Hz)",
    "Low-Mid (200–500 Hz)",
    "Mid (500–2k Hz)",
    "Presence (2k–5k Hz)",
    "High (5k–10k Hz)",
    "Air (10k–16k Hz)",
]

def fmt_mmss(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(round(seconds % 60))
    return f"{m}:{s:02d}"

def safe_db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-12))

def true_peak_db(mono_signal: np.ndarray, sr: int, oversample: int = 4) -> float:
    # Approx true peak by oversampling
    x_os = librosa.resample(mono_signal, orig_sr=sr, target_sr=sr * oversample)
    peak = float(np.max(np.abs(x_os)))
    return safe_db(peak)

def band_balance_percent_db_weighted(y_mono: np.ndarray, sr: int) -> Dict[str, float]:
    """
    "Perceptual-ish" relative band balance.
    We compute average band power -> dB -> softmax with temperature to avoid insane bass dominance.
    """
    n_fft = 4096
    hop = 1024

    S = np.abs(librosa.stft(y_mono, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    nyq = sr / 2.0
    air_top = min(16000.0, nyq)

    bands = {
        "Sub (20–60 Hz)": (20, 60),
        "Low (60–200 Hz)": (60, 200),
        "Low-Mid (200–500 Hz)": (200, 500),
        "Mid (500–2k Hz)": (500, 2000),
        "Presence (2k–5k Hz)": (2000, 5000),
        "High (5k–10k Hz)": (5000, 10000),
        "Air (10k–16k Hz)": (10000, air_top),
    }

    band_power: Dict[str, float] = {}
    for name, (fmin, fmax) in bands.items():
        if fmin >= nyq or fmax <= fmin:
            band_power[name] = 0.0
            continue
        idx = (freqs >= fmin) & (freqs < fmax)
        band_power[name] = float(np.mean(S[idx, :])) if np.any(idx) else 0.0

    db_vals = np.array([10.0 * np.log10(max(v, 1e-20)) for v in band_power.values()], dtype=float)

    temperature = 12.0
    z = (db_vals - np.max(db_vals)) / temperature
    w = np.exp(z)
    w_sum = float(np.sum(w)) + 1e-12
    pct = 100.0 * (w / w_sum)

    return {k: float(pct[i]) for i, k in enumerate(band_power.keys())}

def stereo_width_metrics(y_stereo: np.ndarray, sr: int) -> Tuple[Optional[float], Optional[float]]:
    """
    width_ratio = std(side) / std(mid)
    low_width_ratio under ~120 Hz
    """
    if y_stereo.ndim != 2 or y_stereo.shape[0] != 2:
        return None, None

    L = y_stereo[0, :]
    R = y_stereo[1, :]

    mid = 0.5 * (L + R)
    side = 0.5 * (L - R)
    width_ratio = float(np.std(side) / (np.std(mid) + 1e-12))

    # Low-end width using STFT masking
    n_fft = 4096
    hop = 1024
    SL = librosa.stft(L, n_fft=n_fft, hop_length=hop)
    SR = librosa.stft(R, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_mask = (freqs < 120).astype(float)[:, None]

    L_low = librosa.istft(SL * low_mask, hop_length=hop)
    R_low = librosa.istft(SR * low_mask, hop_length=hop)

    mid_low = 0.5 * (L_low + R_low)
    side_low = 0.5 * (L_low - R_low)
    low_width_ratio = float(np.std(side_low) / (np.std(mid_low) + 1e-12))

    return width_ratio, low_width_ratio

def load_audio(file, mono: bool = False):
    # librosa can read from file-like object Streamlit provides
    y, sr = librosa.load(file, sr=None, mono=mono)
    return y, sr

@dataclass
class AudioMetrics:
    duration_sec: float
    tempo: float
    lufs_i: float
    true_peak_dbtp: float
    band_pct: Dict[str, float]
    width_ratio: Optional[float]
    low_width_ratio: Optional[float]
    sr: int

def compute_metrics(uploaded_file) -> AudioMetrics:
    # Load stereo if available
    y, sr = load_audio(uploaded_file, mono=False)

    if y.ndim == 1:
        y_mono = y
        y_stereo = None
    else:
        y_stereo = y
        y_mono = np.mean(y, axis=0)

    duration_sec = float(len(y_mono) / sr)

    tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    # pyloudnorm expects (samples, channels)
    if y.ndim == 1:
        y_for_ln = np.stack([y_mono, y_mono], axis=1)
    else:
        y_for_ln = y_stereo.T

    meter = pyln.Meter(sr)
    lufs_i = float(meter.integrated_loudness(y_for_ln))

    tp_db = float(true_peak_db(y_mono, sr=sr, oversample=4))
    band_pct = band_balance_percent_db_weighted(y_mono, sr)

    width_ratio, low_width_ratio = (None, None)
    if y_stereo is not None:
        width_ratio, low_width_ratio = stereo_width_metrics(y_stereo, sr)

    return AudioMetrics(
        duration_sec=duration_sec,
        tempo=tempo,
        lufs_i=lufs_i,
        true_peak_dbtp=tp_db,
        band_pct=band_pct,
        width_ratio=width_ratio,
        low_width_ratio=low_width_ratio,
        sr=sr,
    )

def build_llm_stats_text(m: AudioMetrics) -> str:
    width_text = f"{m.width_ratio:.2f}" if m.width_ratio is not None else "N/A"
    low_width_text = f"{m.low_width_ratio:.2f}" if m.low_width_ratio is not None else "N/A"

    return f"""
Audio stats:
- Length: {fmt_mmss(m.duration_sec)}
- Estimated BPM: {m.tempo:.1f}
- LUFS-I: {m.lufs_i:.1f}
- True Peak (dBTP approx): {m.true_peak_dbtp:.1f}
- Band balance % (dB-weighted): Sub20-60={m.band_pct["Sub (20–60 Hz)"]:.1f}, Low60-200={m.band_pct["Low (60–200 Hz)"]:.1f}, LowMid200-500={m.band_pct["Low-Mid (200–500 Hz)"]:.1f}, Mid500-2k={m.band_pct["Mid (500–2k Hz)"]:.1f}, Presence2k-5k={m.band_pct["Presence (2k–5k Hz)"]:.1f}, High5k-10k={m.band_pct["High (5k–10k Hz)"]:.1f}, Air10k-16k={m.band_pct["Air (10k–16k Hz)"]:.1f}
- Stereo width Side/Mid: {width_text}
- Low-end width <120Hz Side/Mid: {low_width_text}
""".strip()

# ----------------------------
# Heuristics: chips, checklist, confidence
# ----------------------------

def chip_for_true_peak(tp_db: float) -> Tuple[str, str, str]:
    # tp_db is dBFS-ish, 0 is clipping risk in many playback chains
    if tp_db >= -0.2:
        return ("Clipping Risk", f"dBTP {tp_db:.1f}", "danger")
    if tp_db >= -1.0:
        return ("Watch True Peak", f"dBTP {tp_db:.1f}", "warn")
    return ("True Peak Safe", f"dBTP {tp_db:.1f}", "good")

def chip_for_lufs(lufs_i: float) -> Tuple[str, str, str]:
    # Genre-agnostic-ish bands for "loud-ish premaster" and "very hot"
    if lufs_i > -8.5:
        return ("Very Hot Loudness", f"LUFS {lufs_i:.1f}", "warn")
    if lufs_i > -11.5:
        return ("Loud-ish Mix", f"LUFS {lufs_i:.1f}", "good")
    return ("Quiet Mix", f"LUFS {lufs_i:.1f}", "warn")

def chip_for_sub_mono(low_width_ratio: Optional[float]) -> Tuple[str, str, str]:
    if low_width_ratio is None:
        return ("Low End Mono", "N/A", "warn")
    if low_width_ratio <= 0.20:
        return ("Sub Mono Safe", f"<120Hz width {low_width_ratio:.2f}", "good")
    if low_width_ratio <= 0.35:
        return ("Sub Width Watch", f"<120Hz width {low_width_ratio:.2f}", "warn")
    return ("Sub Width Risk", f"<120Hz width {low_width_ratio:.2f}", "danger")

def confidence_for_domain(
    domain: str,
    m: AudioMetrics,
    has_ref: bool,
    user_notes_present: bool,
) -> int:
    """
    Simple confidence meter:
    - Based on whether we have supporting metrics for that domain
    - Not a model truth meter, just "how strongly the available data supports this section"
    """
    base = 52

    # Audio always boosts confidence
    base += 18

    # Notes add context (especially arrangement and A&R critique)
    if user_notes_present:
        base += 6

    # Reference improves confidence for balance and mastering comparisons
    if has_ref:
        base += 6

    # Domain-specific boosts based on signal
    if domain == "A&R Critique":
        base += 2 if user_notes_present else 0

    if domain == "Arrangement & Transitions":
        base -= 8
        if user_notes_present:
            base += 4

    if domain == "Low End":
        base += 10
        if m.low_width_ratio is None:
            base -= 6

    if domain == "Mix Balance & Clarity":
        base += 10

    if domain == "Mastering & Loudness":
        base += 10
        if m.true_peak_dbtp >= -1.0:
            base += 2

    if domain == "Reference Match":
        base += 8 if has_ref else -10

    if domain == "Next Actions":
        base += 6

    return int(max(15, min(95, base)))

def build_checklist(m: AudioMetrics) -> pd.DataFrame:
    """
    Pass, Watch, Fail checks (heuristic).
    """
    rows = []

    # True peak
    if m.true_peak_dbtp >= -0.2:
        status = "Fail"
        note = "True peak is extremely hot. Set limiter ceiling around -1.0 dBTP."
    elif m.true_peak_dbtp >= -1.0:
        status = "Watch"
        note = "True peak is close to playback risk. Consider -1.0 dBTP ceiling."
    else:
        status = "Pass"
        note = "True peak looks safe for typical playback chains."

    rows.append(("True Peak Headroom", status, note))

    # Loudness
    if m.lufs_i > -8.5:
        status = "Watch"
        note = "Very hot loudness. Check punch and transient integrity."
    elif m.lufs_i > -11.5:
        status = "Pass"
        note = "Reasonable loud-ish level for electronic. Still watch true peak."
    else:
        status = "Watch"
        note = "Quiet mix. You may struggle to compete without rebalancing dynamics."

    rows.append(("Loudness (LUFS-I)", status, note))

    # Sub mono
    if m.low_width_ratio is None:
        status = "Watch"
        note = "Mono file or unable to calculate. Verify sub is mono below 120 Hz."
    elif m.low_width_ratio <= 0.20:
        status = "Pass"
        note = "Sub appears mono-safe."
    elif m.low_width_ratio <= 0.35:
        status = "Watch"
        note = "Some sub width. Check mono collapse on a mono sum."
    else:
        status = "Fail"
        note = "High sub width. Likely collapses in mono and loses punch."

    rows.append(("Sub Mono Compatibility", status, note))

    # Low-mid risk
    lowmid = m.band_pct["Low-Mid (200–500 Hz)"]
    if lowmid >= 20.0:
        status = "Watch"
        note = "Low-mids look elevated. Likely masking kick attack and clarity."
    else:
        status = "Pass"
        note = "Low-mid balance looks reasonable relative to other bands."

    rows.append(("Low-Mid Masking Risk", status, note))

    # Air/presence
    presence = m.band_pct["Presence (2k–5k Hz)"]
    air = m.band_pct["Air (10k–16k Hz)"]
    if (presence + air) < 5.0:
        status = "Watch"
        note = "Top-end energy looks low. May feel dull or lack excitement."
    else:
        status = "Pass"
        note = "Top-end presence looks present enough for translation."

    rows.append(("Top-End Excitement", status, note))

    df = pd.DataFrame(rows, columns=["Check", "Status", "Notes"])
    return df

# ----------------------------
# LLM prompts and parsing
# ----------------------------

def system_prompt_for_tab(tab_name: str) -> str:
    common_rules = """
You are Producer Co-Pilot (by Taha Asadi), an elite production mentor operating at label-ready level.
You think like a seasoned producer, mix engineer, and A&R combined.

IMPORTANT RULES:
- Use TIMESTAMPS (mm:ss) and relative positions (intro, first drop, breakdown, second drop), NOT bar counts unless the user explicitly asks for bar math.
- Base your advice on professional translation: loudness, dynamics, low-end, stereo/mono compatibility, and arrangement energy.
- Respect the track length when proposing timestamps.
- If audio stats are missing, be transparent and use notes only.
- Be specific and technical. Avoid generic beginner advice.
- Explain why recommendations work on a dance floor (tension, expectation, contrast, impact), but do not use the word "club" excessively.
""".strip()

    tab_rules = {
        "A&R Critique": """
Output style:
- Short, decisive, A&R style.
- Call out the biggest 3 issues and biggest 2 strengths.
- Finish with a "Decision" line: "Sign-ready", "Close but needs work", or "Not ready".

Return:
- A short critique, then a tight prioritized list of fixes.
""".strip(),
        "Arrangement & Transitions": """
Focus:
- Energy curve, contrast, tension, drop impact, breakdown pacing.
- Give timestamp-specific actions for transitions.

Return:
- A sectioned blueprint with timestamps, then 5 transition upgrades.
""".strip(),
        "Low End": """
Focus:
- Kick vs bass separation, sub management, mono stability, sidechain movement, transient integrity.

Return:
- Diagnose, then prescribe: EQ moves, dynamic moves, sidechain timing, mono strategy.
""".strip(),
        "Mix Balance & Clarity": """
Focus:
- Low-mid masking, mid presence, harshness, depth, stereo placement by band.

Return:
- A "What to fix first" list, then concrete EQ/dynamics suggestions with ranges.
""".strip(),
        "Mastering & Loudness": """
Focus:
- LUFS strategy, true peak safety, punch preservation, limiter approach, clipping strategy.

Return:
- A safe loudness path, suggested limiter ceiling, and 3 mastering chain options (conceptual).
""".strip(),
        "Next Actions": """
Focus:
- Concrete DAW actions you can do in 60 minutes.

Return:
- A 10-step checklist, each step should be one action with a target and why.
""".strip(),
    }

    return f"{common_rules}\n\n{tab_rules.get(tab_name, '')}".strip()

def build_user_content(notes: str, stats_text: str, style: str, has_ref: bool, ref_delta_text: str) -> str:
    extra = ""
    if has_ref and ref_delta_text:
        extra = f"\n\nReference comparison (track vs reference):\n{ref_delta_text}".strip()

    return f"""
Target style:
{style}

User notes:
{notes}

{stats_text}
{extra}
""".strip()

def call_llm(client: OpenAI, tab_name: str, user_content: str) -> str:
    with st.spinner(f"Generating: {tab_name}..."):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt_for_tab(tab_name)},
                {"role": "user", "content": user_content},
            ],
        )
    return resp.choices[0].message.content

# ----------------------------
# Reference match utilities
# ----------------------------

def delta_summary(main: AudioMetrics, ref: AudioMetrics) -> str:
    """
    Small comparison text for the LLM, plus the UI can show a table.
    """
    def d(a, b) -> float:
        return float(a - b)

    width_main = main.width_ratio if main.width_ratio is not None else np.nan
    width_ref = ref.width_ratio if ref.width_ratio is not None else np.nan
    loww_main = main.low_width_ratio if main.low_width_ratio is not None else np.nan
    loww_ref = ref.low_width_ratio if ref.low_width_ratio is not None else np.nan

    bands = list(main.band_pct.keys())
    band_lines = []
    for k in bands:
        band_lines.append(f"- {k}: {d(main.band_pct[k], ref.band_pct.get(k, 0.0)):+.1f} pts")

    return f"""
Key deltas (Main minus Reference):
- LUFS-I: {d(main.lufs_i, ref.lufs_i):+.1f}
- True Peak (dBTP): {d(main.true_peak_dbtp, ref.true_peak_dbtp):+.1f}
- Stereo width (Side/Mid): {d(width_main, width_ref):+.2f}
- Low-end width <120Hz: {d(loww_main, loww_ref):+.2f}
Band balance deltas:
{chr(10).join(band_lines)}
""".strip()

def reference_table(main: AudioMetrics, ref: AudioMetrics) -> pd.DataFrame:
    rows = []
    rows.append(("Length", fmt_mmss(main.duration_sec), fmt_mmss(ref.duration_sec), ""))
    rows.append(("BPM (est)", f"{main.tempo:.1f}", f"{ref.tempo:.1f}", f"{(main.tempo-ref.tempo):+.1f}"))
    rows.append(("LUFS-I", f"{main.lufs_i:.1f}", f"{ref.lufs_i:.1f}", f"{(main.lufs_i-ref.lufs_i):+.1f}"))
    rows.append(("True Peak (dBTP)", f"{main.true_peak_dbtp:.1f}", f"{ref.true_peak_dbtp:.1f}", f"{(main.true_peak_dbtp-ref.true_peak_dbtp):+.1f}"))

    wm = main.width_ratio if main.width_ratio is not None else np.nan
    wr = ref.width_ratio if ref.width_ratio is not None else np.nan
    rows.append((
        "Stereo Width (Side/Mid)",
        f"{wm:.2f}" if np.isfinite(wm) else "N/A",
        f"{wr:.2f}" if np.isfinite(wr) else "N/A",
        f"{(wm-wr):+.2f}" if np.isfinite(wm) and np.isfinite(wr) else ""
    ))

    lm = main.low_width_ratio if main.low_width_ratio is not None else np.nan
    lr = ref.low_width_ratio if ref.low_width_ratio is not None else np.nan
    rows.append((
        "Low-End Width <120Hz",
        f"{lm:.2f}" if np.isfinite(lm) else "N/A",
        f"{lr:.2f}" if np.isfinite(lr) else "N/A",
        f"{(lm-lr):+.2f}" if np.isfinite(lm) and np.isfinite(lr) else ""
    ))

    for k in main.band_pct.keys():
        mv = main.band_pct[k]
        rv = ref.band_pct.get(k, np.nan)
        rows.append((f"Band: {k}", f"{mv:.1f}%", f"{rv:.1f}%" if np.isfinite(rv) else "N/A", f"{(mv-rv):+.1f} pts" if np.isfinite(rv) else ""))

    return pd.DataFrame(rows, columns=["Metric", "Main", "Reference", "Delta"])

# ----------------------------
# App setup
# ----------------------------

load_dotenv()
inject_css()

st.markdown(
    """
    <div class="pc-header">
      <div class="pc-title">Producer Co-Pilot 🎛️</div>
      <div class="pc-subtitle">Upload a mixdown. Run one analysis. Get a production breakdown you can actually use.</div>
      <div class="pc-subtitle" style="margin-top:10px;">
        <a href="https://www.tahaasadi.com" target="_blank" style="font-weight:800; text-decoration:none;">
          Producer Co-Pilot by Taha Asadi
        </a>.
        Metrics are heuristic. Always let your ears decide.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# OpenAI key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY. Create a .env file with OPENAI_API_KEY=YOUR_KEY")
    st.stop()

client = OpenAI(api_key=api_key)

# Session state
if "main_metrics" not in st.session_state:
    st.session_state.main_metrics = None
if "ref_metrics" not in st.session_state:
    st.session_state.ref_metrics = None
if "outputs" not in st.session_state:
    st.session_state.outputs = {}
if "stats_text" not in st.session_state:
    st.session_state.stats_text = ""
if "ref_delta_text" not in st.session_state:
    st.session_state.ref_delta_text = ""
if "ran" not in st.session_state:
    st.session_state.ran = False

# ----------------------------
# Sidebar: single "Run Analysis" flow
# ----------------------------

st.sidebar.header("Session")

# Target style list (alphabetical, as requested)
STYLE_OPTIONS = sorted([
    "Ambient / Chill Out",
    "Dance / Pop",
    "Drum & Bass"
    "Dubstep",
    "Electronica",
    "House",
    "Mainstage",
    "Progressive",
    "Synthwave",
    "Techno",
    "Trance",
    "Trap",
])

style_choice = st.sidebar.selectbox(
    "Target Style",
    STYLE_OPTIONS,
    index=STYLE_OPTIONS.index("Electronica") if "Electronica" in STYLE_OPTIONS else 0
)

main_file = st.sidebar.file_uploader("Upload WAV (Main)", type=["wav"], key="main_uploader")

# Reference match mode (optional)
st.sidebar.markdown("### Reference Match Mode (Optional)")
enable_ref = st.sidebar.toggle("Enable Reference Match", value=False)
ref_file = None
if enable_ref:
    ref_file = st.sidebar.file_uploader("Upload WAV (Reference)", type=["wav"], key="ref_uploader")

# Optional notes (optional)
st.sidebar.markdown("### Optional Notes (Optional)")
enable_notes = st.sidebar.toggle("Enable Notes", value=False)
notes_text = ""
if enable_notes:
    notes_text = st.sidebar.text_area(
        "References, pain points, targets, what feels wrong",
        height=110,
        placeholder="Example: target 140 BPM, Afterlife vibe, drop feels weak, low end muddy, breakdown needs more contrast...",
    )
else:
    notes_text = ""

# Single CTA
can_run = main_file is not None
run_clicked = st.sidebar.button("Run Analysis", type="primary", disabled=not can_run)

# If clicked: compute metrics, then generate all tab outputs once
if run_clicked:
    st.session_state.outputs = {}
    st.session_state.ran = False
    st.session_state.ref_metrics = None
    st.session_state.ref_delta_text = ""

    # Analyze main
    with st.spinner("Analyzing main track..."):
        st.session_state.main_metrics = compute_metrics(main_file)
        st.session_state.stats_text = build_llm_stats_text(st.session_state.main_metrics)

    # Analyze ref if enabled
    if enable_ref and ref_file is not None:
        with st.spinner("Analyzing reference track..."):
            st.session_state.ref_metrics = compute_metrics(ref_file)
            st.session_state.ref_delta_text = delta_summary(st.session_state.main_metrics, st.session_state.ref_metrics)

    # Notes fallback (A&R style default)
    final_notes = notes_text.strip()
    if not final_notes:
        final_notes = "Give me a full A&R style critique and a producer action plan. Be practical and specific."

    # Build shared user content
    user_content_common = build_user_content(
        notes=final_notes,
        stats_text=st.session_state.stats_text,
        style=style_choice,
        has_ref=(st.session_state.ref_metrics is not None),
        ref_delta_text=st.session_state.ref_delta_text,
    )

    # Generate tab outputs (one pass)
    tab_names = [
        "A&R Critique",
        "Arrangement & Transitions",
        "Low End",
        "Mix Balance & Clarity",
        "Mastering & Loudness",
        "Next Actions",
    ]

    for t in tab_names:
        st.session_state.outputs[t] = call_llm(client, t, user_content_common)

    st.session_state.ran = True
    st.sidebar.success("Done. Open the tabs to review each section.")

# ----------------------------
# Main area: Tabs
# ----------------------------

tabs = st.tabs([
    "Overview",
    "Professional Checklist",
    "Reference Match",
    "A&R Critique",
    "Arrangement & Transitions",
    "Low End",
    "Mix Balance & Clarity",
    "Mastering & Loudness",
    "Next Actions",
])

# Convenience vars
m: Optional[AudioMetrics] = st.session_state.main_metrics
refm: Optional[AudioMetrics] = st.session_state.ref_metrics
has_ref = refm is not None
user_notes_present = bool(notes_text.strip())

def render_confidence(tab_label: str) -> None:
    if not m:
        st.caption("Confidence Meter: N/A (upload and run analysis)")
        return
    score = confidence_for_domain(tab_label, m, has_ref=has_ref, user_notes_present=user_notes_present)
    st.caption(f"Confidence Meter: {score}/100")
    st.progress(score / 100.0)

def require_run_message() -> None:
    st.info("Run Analysis from the left sidebar to generate your outputs.", icon="ℹ️")

# ----------------------------
# Overview tab
# ----------------------------
with tabs[0]:
    st.subheader("Overview")
    if not m:
        st.write("Upload a WAV and click **Run Analysis** to see metrics, checklist, and outputs.")
    else:
        # Chips
        t_title, t_val, t_sev = chip_for_true_peak(m.true_peak_dbtp)
        l_title, l_val, l_sev = chip_for_lufs(m.lufs_i)
        s_title, s_val, s_sev = chip_for_sub_mono(m.low_width_ratio)

        def chip_html(title: str, val: str, sev: str) -> str:
            cls = {"danger": "pc-chip-danger", "warn": "pc-chip-warn", "good": "pc-chip-good"}.get(sev, "")
            return f'<span class="pc-chip {cls}">{title}: {val}</span>'

        st.markdown(
            chip_html(t_title, t_val, t_sev)
            + chip_html(l_title, l_val, l_sev)
            + chip_html(s_title, s_val, s_sev),
            unsafe_allow_html=True,
        )

        # Metric tiles
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="pc-tile">
                  <div class="pc-tile-label">Length</div>
                  <div class="pc-tile-value">{fmt_mmss(m.duration_sec)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="pc-tile">
                  <div class="pc-tile-label">BPM (est)</div>
                  <div class="pc-tile-value">{m.tempo:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="pc-tile">
                  <div class="pc-tile-label">LUFS-I</div>
                  <div class="pc-tile-value">{m.lufs_i:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""
                <div class="pc-tile">
                  <div class="pc-tile-label">True Peak (dBTP)</div>
                  <div class="pc-tile-value">{m.true_peak_dbtp:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # Rough meters (normalized bars)
        st.caption("Meters (rough)")
        loud_norm = float(np.clip(((-m.lufs_i) - 6.0) / 10.0, 0.0, 1.0))  # -6 very loud, -16 quiet
        tp_norm = float(np.clip((m.true_peak_dbtp + 6.0) / 6.0, 0.0, 1.0))  # -6..0

        st.write("Loudness (higher bar means louder)")
        st.progress(loud_norm)
        st.write("True Peak (higher bar means closer to 0 dBTP)")
        st.progress(tp_norm)

        st.divider()

        # Balance chart (Sub first)
        st.subheader("Balance By Band (Relative)")
        band_labels = BAND_ORDER
        band_vals = [m.band_pct.get(k, 0.0) for k in band_labels]
        df_band = pd.DataFrame({"Band": band_labels, "Relative Energy (%)": band_vals}).set_index("Band")
        st.bar_chart(df_band)

        st.divider()

        st.subheader("Stereo")
        width_text = f"{m.width_ratio:.2f}" if m.width_ratio is not None else "N/A"
        low_width_text = f"{m.low_width_ratio:.2f}" if m.low_width_ratio is not None else "N/A"
        st.write(f"Stereo width (Side/Mid): **{width_text}**")
        st.write(f"Low-end width <120 Hz (Side/Mid): **{low_width_text}**")

        st.markdown(
            """
            <div class="pc-footer">
              Producer Co-Pilot by <a href="https://www.tahaasadi.com" target="_blank">Taha Asadi</a>.
              Metrics are heuristic. Always let your ears decide.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------------
# Professional Checklist tab
# ----------------------------
with tabs[1]:
    st.subheader("Professional Translation Checklist")
    if not m:
        require_run_message()
    else:
        render_confidence("Mix Balance & Clarity")
        df = build_checklist(m)

        def status_color(s: str) -> str:
            if s == "Pass":
                return "✅ Pass"
            if s == "Watch":
                return "⚠️ Watch"
            return "🛑 Fail"

        df_show = df.copy()
        df_show["Status"] = df_show["Status"].apply(status_color)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        st.caption("Tip: Use this as a fast pre-submit gate before you print a premaster or send a demo.")

# ----------------------------
# Reference Match tab
# ----------------------------
with tabs[2]:
    st.subheader("Reference Match (Optional)")
    if not m:
        require_run_message()
    else:
        render_confidence("Reference Match")

        if not has_ref:
            st.info("Enable Reference Match in the sidebar and upload a reference WAV to compare.", icon="ℹ️")
        else:
            df_ref = reference_table(m, refm)
            st.dataframe(df_ref, use_container_width=True, hide_index=True)

            st.divider()

            st.subheader("Interpretation")
            st.write(
                "Deltas are **Main minus Reference**. Use this to align tonal balance, loudness strategy, and stereo behavior without copying the reference."
            )

# ----------------------------
# Output tabs
# ----------------------------

def render_output_tab(tab_label: str, key: str) -> None:
    st.subheader(tab_label)
    if not m or not st.session_state.ran:
        require_run_message()
        return

    render_confidence(tab_label)

    out = st.session_state.outputs.get(key, "")
    if not out:
        st.warning("No output found for this section. Re-run analysis.", icon="⚠️")
        return

    st.markdown(out)

with tabs[3]:
    render_output_tab("Full Critique (A&R Style)", "A&R Critique")

with tabs[4]:
    render_output_tab("Arrangement & Transitions", "Arrangement & Transitions")

with tabs[5]:
    render_output_tab("Low End", "Low End")

with tabs[6]:
    render_output_tab("Mix Balance & Clarity", "Mix Balance & Clarity")

with tabs[7]:
    render_output_tab("Mastering & Loudness", "Mastering & Loudness")

with tabs[8]:
    render_output_tab("Next Actions", "Next Actions")