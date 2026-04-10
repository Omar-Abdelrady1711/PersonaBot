"""
app.py  —  Persona-Maintenance Dialogue System
Run:  streamlit run app.py
"""

import streamlit as st
import math, random

MOCK_MODE = False
from dialogue_engine import get_bot_response, BASELINE_PROMPTS
from persona_schema   import Persona, PRESETS
from alignment_scorer import score_alignment
from reinforcement    import score_and_reinforce

st.set_page_config(page_title="PersonaBot", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family:'Inter',sans-serif !important; font-size:16px !important; }
h1 { font-family:'Syne',sans-serif !important; font-size:2.4rem !important; font-weight:800 !important; letter-spacing:-0.03em !important; margin-bottom:0.1rem !important; }
.bubble-user { background:#1a2035; border-left:3px solid #4f8ef7; padding:14px 18px; border-radius:8px; margin:10px 0 4px; font-size:1rem; line-height:1.7; color:#dce3f0; }
.bubble-bot  { background:#111825; border-left:3px solid #3dd68c; padding:14px 18px; border-radius:8px; margin:4px 0 10px; font-size:1rem; line-height:1.7; color:#dce3f0; }
.bubble-label { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:8px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.lbl-you { color:#4f8ef7; } .lbl-bot { color:#3dd68c; }
.chip { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; padding:2px 9px; border-radius:4px; font-weight:500; }
.chip-blue  { background:#0d1e3a; color:#4f8ef7; border:1px solid #1e3a6a; }
.chip-green { background:#0a2018; color:#3dd68c; border:1px solid #1a4030; }
.chip-amber { background:#1e1400; color:#f0a030; border:1px solid #3a2800; }
.chip-red   { background:#1e0a0a; color:#e05555; border:1px solid #3a1010; }
.corr-box { background:#180808; border-left:3px solid #e05555; padding:10px 14px; border-radius:6px; margin-top:8px; font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#cc5555; line-height:1.5; }
.trow  { display:flex; align-items:center; gap:10px; margin:5px 0; }
.tname { font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#8898bb; width:80px; flex-shrink:0; }
.tbar  { flex:1; height:4px; background:#1c2a40; border-radius:2px; overflow:hidden; }
.tfill { height:100%; background:#4f8ef7; border-radius:2px; }
.tval  { font-family:'IBM Plex Mono',monospace; font-size:0.66rem; color:#4a5570; width:30px; text-align:right; }
.scard { background:#0e1525; border:1px solid #1c2a40; border-radius:8px; padding:14px 18px; margin:5px 0; }
.sval  { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; line-height:1; }
.slbl  { font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:#5a6888; letter-spacing:0.08em; text-transform:uppercase; margin-top:4px; }
.avgcard { border-radius:8px; padding:14px 18px; margin:0; border-left:3px solid; }
.ccard { background:#0e1525; border:1px solid #1c2a40; border-radius:8px; padding:14px 16px; }
.clabel { font-family:'IBM Plex Mono',monospace; font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase; color:#5a6888; margin-bottom:10px; display:flex; align-items:center; gap:8px; }
.cuser  { background:#12192a; border-left:2px solid #4f8ef7; padding:8px 12px; border-radius:6px; margin:8px 0 4px; font-size:0.86rem; color:#8898bb; }
.cbot   { background:#0c1420; border-left:2px solid #3dd68c; padding:10px 12px; border-radius:6px; margin:4px 0 8px; font-size:0.86rem; line-height:1.65; color:#c8d4e8; }
.sec { font-family:'IBM Plex Mono',monospace; font-size:0.66rem; letter-spacing:0.12em; text-transform:uppercase; color:#3a4a60; border-bottom:1px solid #1c2a40; padding-bottom:5px; margin-bottom:10px; }
.persona-desc { font-size:0.82rem; color:#6a7a99; line-height:1.5; padding:10px 14px; background:#0a1628; border:1px solid #1e3a6a; border-radius:8px; margin-bottom:10px; }
[data-testid="stSidebar"] { border-right:1px solid #1c2a40 !important; }
#MainMenu, footer { visibility:hidden; }
header { visibility:visible !important; }
.block-container { padding-top:1.5rem !important; }
</style>
""", unsafe_allow_html=True)

TRAITS = ["formality","empathy","technical_depth","verbosity",
          "assertiveness","humor","politeness","curiosity"]
TLBLS  = {"formality":"Formality","empathy":"Empathy","technical_depth":"Tech Depth",
           "verbosity":"Verbosity","assertiveness":"Assert.","humor":"Humor",
           "politeness":"Politeness","curiosity":"Curiosity"}
PERSONA_DESCRIPTIONS = {
    "formal_expert":       "Professional, technical, and precise. Formal vocabulary throughout.",
    "friendly_support":    "Warm, empathetic, and encouraging. Always acknowledges feelings first.",
    "casual_tutor":        "Relaxed and playful. Uses analogies and asks follow-up questions.",
    "analytical_assistant":"Objective and data-driven. Minimal emotion, maximum clarity.",
}

def _mock_resp(msg, hist, persona, correction, cond):
    _MOCK = {
        "formal_expert":       "The subject matter pertains to a well-documented domain. Permit me to elucidate with precision and rigour.",
        "friendly_support":    "I totally understand how you're feeling — that sounds really tough! You're not alone in this. What's been the hardest part?",
        "casual_tutor":        "Okay so basically — it's pretty neat once it clicks! Think of it like this… does that make sense so far?",
        "analytical_assistant":"Three primary factors identified. Recommendation follows logically from available data.",
    }
    base = _MOCK.get(persona.name if persona else "formal_expert", "...")
    if cond == "baseline":
        base = base.replace("Permit me to elucidate", "Let me explain").replace("pertains", "is about")
    if correction and cond == "adaptive_reinforce":
        base = "[Recalibrated] " + base
    preset = f"{persona.name} (baseline)" if cond == "baseline" else (persona.name if persona else "none")
    return dict(user_message=msg, bot_response=base, detected_persona=persona,
                active_preset=preset, alignment_score=None,
                reinforcement_applied=False, correction=correction)

run_resp = lambda *a: _mock_resp(*a) if MOCK_MODE else get_bot_response(*a)

for k, v in dict(
    hist_base=[], hist_adap=[], hist_reinf=[],
    selected_preset="formal_expert",
    pending_correction=None,
    turn_count=0,
    cmp_hist_base=[], cmp_hist_adap=[], cmp_hist_reinf=[],
    cmp_pending_correction=None,
).items():
    if k not in st.session_state:
        st.session_state[k] = v

def sc(v): return "chip-green" if v >= 0.72 else ("chip-amber" if v >= 0.52 else "chip-red")
def sc_color(v): return "#3dd68c" if v >= 0.72 else ("#f0a030" if v >= 0.52 else "#e05555")

def trait_bars(persona):
    html = ""
    for t in TRAITS:
        val = getattr(persona, t); pct = int(val * 100)
        html += (f'<div class="trow"><span class="tname">{TLBLS[t]}</span>'
                 f'<div class="tbar"><div class="tfill" style="width:{pct}%"></div></div>'
                 f'<span class="tval">{val:.2f}</span></div>')
    return html

def radar(scores, persona, sz=220):
    n = len(TRAITS); cx = cy = sz // 2; r = sz // 2 - 36
    short = ["Form","Emp","Tech","Verb","Asrt","Hmr","Pol","Cur"]
    def pt(i, v):
        a = math.pi / 2 + 2 * math.pi * i / n
        return cx + r * v * math.cos(a), cy - r * v * math.sin(a)
    grid = "".join(f'<polygon points="{" ".join(f"{pt(i,lv)[0]:.1f},{pt(i,lv)[1]:.1f}" for i in range(n))}" fill="none" stroke="#1c2a40" stroke-width="1"/>' for lv in [.25,.5,.75,1.])
    axes = "".join(f'<line x1="{cx}" y1="{cy}" x2="{pt(i,1)[0]:.1f}" y2="{pt(i,1)[1]:.1f}" stroke="#1c2a40" stroke-width="1"/>' for i in range(n))
    tp   = " ".join(f"{pt(i,getattr(persona,TRAITS[i]))[0]:.1f},{pt(i,getattr(persona,TRAITS[i]))[1]:.1f}" for i in range(n))
    tgt  = f'<polygon points="{tp}" fill="#4f8ef710" stroke="#4f8ef7" stroke-width="1.5" stroke-dasharray="4,3"/>'
    sv   = [scores.get(t,.5) for t in TRAITS]
    sp   = " ".join(f"{pt(i,sv[i])[0]:.1f},{pt(i,sv[i])[1]:.1f}" for i in range(n))
    scr  = f'<polygon points="{sp}" fill="#3dd68c18" stroke="#3dd68c" stroke-width="2"/>'
    dots = "".join(f'<circle cx="{pt(i,sv[i])[0]:.1f}" cy="{pt(i,sv[i])[1]:.1f}" r="3" fill="#3dd68c"/>' for i in range(n))
    lbls = ""
    for i, lbl in enumerate(short):
        lx, ly = pt(i, 1.38); anc = "middle"
        if lx < cx - 8: anc = "end"
        elif lx > cx + 8: anc = "start"
        lbls += f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anc}" font-size="9" fill="#6a7a99" font-family="IBM Plex Mono,monospace">{lbl}</text>'
    leg = (f'<rect x="6" y="{sz-20}" width="10" height="2" fill="#4f8ef7" opacity=".7"/>'
           f'<text x="20" y="{sz-15}" font-size="8" fill="#5a6888" font-family="IBM Plex Mono,monospace">target</text>'
           f'<rect x="6" y="{sz-12}" width="10" height="2" fill="#3dd68c"/>'
           f'<text x="20" y="{sz-7}" font-size="8" fill="#5a6888" font-family="IBM Plex Mono,monospace">response</text>')
    return (f'<svg width="{sz}" height="{sz}" xmlns="http://www.w3.org/2000/svg" style="border-radius:8px;display:block">'
            f'{grid}{axes}{tgt}{scr}{dots}{lbls}{leg}</svg>')


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — clean and minimal
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sec">Persona</div>', unsafe_allow_html=True)
    selected = st.selectbox("",
        options=list(PRESETS.keys()),
        format_func=lambda x: {
            "formal_expert":       "🎓  Formal Expert",
            "friendly_support":    "🤗  Friendly Support",
            "casual_tutor":        "📚  Casual Tutor",
            "analytical_assistant":"🔬  Analytical Assistant",
        }[x],
        label_visibility="collapsed",
        key="selected_preset",
    )
    persona = PRESETS[selected]
    st.markdown(f'<div class="persona-desc">{PERSONA_DESCRIPTIONS[selected]}</div>', unsafe_allow_html=True)
    st.markdown(trait_bars(persona), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">Condition</div>', unsafe_allow_html=True)
    condition = st.radio("",
        ["baseline", "adaptive", "adaptive_reinforce"],
        format_func=lambda x: {
            "baseline":           "① Baseline",
            "adaptive":           "② Persona-Controlled",
            "adaptive_reinforce": "③ Persona + Reinforce",
        }[x],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">Threshold</div>', unsafe_allow_html=True)
    threshold = st.slider("", 0.3, 0.9, 0.55, 0.05, label_visibility="collapsed")
    st.caption(f"Reinforcement fires when alignment < {threshold:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↺  Reset All", use_container_width=True):
        for k in ["hist_base","hist_adap","hist_reinf",
                  "cmp_hist_base","cmp_hist_adap","cmp_hist_reinf"]:
            st.session_state[k] = []
        st.session_state.pending_correction = None
        st.session_state.cmp_pending_correction = None
        st.session_state.turn_count = 0
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
hkey    = {"baseline":"hist_base","adaptive":"hist_adap","adaptive_reinforce":"hist_reinf"}[condition]
history = st.session_state[hkey]
persona = PRESETS[st.session_state.selected_preset]

st.markdown("# 🧠 PersonaBot")
st.caption("Persona-Adaptive Dialogue System  ·  NLP Course Project  ·  April 2026")
if MOCK_MODE:
    st.warning("⚠ MOCK MODE — set `MOCK_MODE = False` to use the live engine", icon="⚠️")
st.divider()

chat_col, panel_col = st.columns([3, 1], gap="large")

# ── Single-condition chat ─────────────────────────────────────────────────────
with chat_col:
    clabels = {
        "baseline":           "① Baseline",
        "adaptive":           "② Persona-Controlled",
        "adaptive_reinforce": "③ Persona + Reinforcement",
    }
    st.markdown(f'<div class="sec">{clabels[condition]}</div>', unsafe_allow_html=True)

    for turn in history:
        st.markdown(
            f'<div class="bubble-user"><div class="bubble-label lbl-you">You</div>{turn["user_message"]}</div>',
            unsafe_allow_html=True)
        scores  = turn.get("alignment_score") or {}
        overall = scores.get("overall")
        score_chip = f'<span class="chip {sc(overall)}">{overall:.2f}</span>' if overall is not None else ""
        reinf_chip = '<span class="chip chip-red">🔧 corrected</span>' if turn.get("reinforcement_applied") else ""
        corr_html  = f'<div class="corr-box">⚡ {turn["correction"]}</div>' if turn.get("correction") else ""
        st.markdown(
            f'<div class="bubble-bot"><div class="bubble-label lbl-bot">{score_chip} {reinf_chip}</div>'
            f'{turn["bot_response"]}{corr_html}</div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            user_input = st.text_input("", "", placeholder="Type your message…", label_visibility="collapsed")
        with c2:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and user_input.strip():
        with st.spinner("Thinking…"):
            pending = st.session_state.pending_correction if condition == "adaptive_reinforce" else None
            turn = run_resp(user_input, history, persona, pending, condition)
            if condition in ("adaptive", "adaptive_reinforce"):
                turn = score_and_reinforce(turn, threshold=threshold)
                if condition == "adaptive":
                    turn["correction"] = None
                    turn["reinforcement_applied"] = False
                st.session_state.pending_correction = turn.get("correction") if condition == "adaptive_reinforce" else None
            else:
                turn["alignment_score"] = score_alignment(turn["bot_response"], persona)
                turn["reinforcement_applied"] = False
                turn["correction"] = None
                st.session_state.pending_correction = None
            st.session_state[hkey].append(turn)
            st.session_state.turn_count += 1
        st.rerun()

# ── Right panel ───────────────────────────────────────────────────────────────
with panel_col:
    st.markdown('<div class="sec">Radar</div>', unsafe_allow_html=True)
    if history:
        scores = history[-1].get("alignment_score") or {}
        if scores:
            st.markdown(radar(scores, persona), unsafe_allow_html=True)
        else:
            st.caption("No data yet.")
    else:
        st.caption("Send a message to see the radar.")

    if history:
        scores = history[-1].get("alignment_score") or {}
        if scores:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sec">Last Turn</div>', unsafe_allow_html=True)
            rows = ""
            for t in ["formality","empathy","technical_depth","verbosity","politeness","curiosity"]:
                v = scores.get(t, 0.)
                rows += (f'<div style="display:flex;justify-content:space-between;align-items:center;'
                         f'padding:3px 0;border-bottom:1px solid #141e2e">'
                         f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#7a8aaa">{TLBLS[t]}</span>'
                         f'<span class="chip {sc(v)}" style="font-size:0.72rem">{v:.2f}</span></div>')
            ov = scores.get("overall", 0.)
            rows += (f'<div style="display:flex;justify-content:space-between;align-items:center;padding:7px 0 0">'
                     f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#c8d4e8;font-weight:600">Overall</span>'
                     f'<span class="chip {sc(ov)}" style="font-size:0.8rem;padding:3px 10px">{ov:.2f}</span></div>')
            st.markdown(rows, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">Stats</div>', unsafe_allow_html=True)
    reinf = sum(1 for t in history if t.get("reinforcement_applied"))
    avgs  = [t["alignment_score"]["overall"] for t in history if isinstance(t.get("alignment_score"), dict)]
    avg   = sum(avgs) / len(avgs) if avgs else 0.
    st.markdown(f"""
    <div class="scard"><div class="sval">{len(history)}</div><div class="slbl">Turns</div></div>
    <div class="scard"><div class="sval" style="color:#e05555">{reinf}</div><div class="slbl">Corrections</div></div>
    <div class="scard"><div class="sval" style="color:{sc_color(avg)}">{avg:.2f}</div><div class="slbl">Avg Alignment</div></div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-TURN SIDE-BY-SIDE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### Side-by-Side Comparison")
st.caption(f"All three conditions receive the same messages targeting `{st.session_state.selected_preset}`.")

cmp_hists = {
    "baseline":           st.session_state.cmp_hist_base,
    "adaptive":           st.session_state.cmp_hist_adap,
    "adaptive_reinforce": st.session_state.cmp_hist_reinf,
}
cmp_configs = [
    ("baseline",           "① Baseline",             "#4a5570"),
    ("adaptive",           "② Persona-Controlled",   "#4f8ef7"),
    ("adaptive_reinforce", "③ Persona + Reinforce",  "#3dd68c"),
]

# Avg alignment summary
if any(len(h) > 0 for h in cmp_hists.values()):
    stat_cols = st.columns(3, gap="medium")
    for stat_col, (cond, label, color) in zip(stat_cols, cmp_configs):
        hist = cmp_hists[cond]
        avgs = [t["alignment_score"]["overall"] for t in hist if isinstance(t.get("alignment_score"), dict)]
        avg  = sum(avgs) / len(avgs) if avgs else 0.
        with stat_col:
            st.markdown(
                f'<div class="scard" style="border-left:3px solid {color}">'
                f'<div class="sval" style="color:{color}">{avg:.2f}</div>'
                f'<div class="slbl">{label}</div></div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Conversation turns
if any(len(h) > 0 for h in cmp_hists.values()):
    max_turns = max(len(h) for h in cmp_hists.values())
    for turn_idx in range(max_turns):
        cols = st.columns(3, gap="medium")
        for col, (cond, label, color) in zip(cols, cmp_configs):
            hist = cmp_hists[cond]
            with col:
                if turn_idx < len(hist):
                    turn    = hist[turn_idx]
                    scores  = turn.get("alignment_score") or {}
                    overall = scores.get("overall")
                    score_chip = f'<span class="chip {sc(overall)}">{overall:.2f}</span>' if overall is not None else ""
                    reinf_chip = '<span class="chip chip-red">🔧</span>' if turn.get("reinforcement_applied") else ""
                    corr_html  = f'<div class="corr-box" style="font-size:0.7rem">⚡ {turn["correction"]}</div>' if turn.get("correction") else ""
                    st.markdown(
                        f'<div class="ccard">'
                        f'<div class="clabel"><span style="width:7px;height:7px;border-radius:50%;background:{color};display:inline-block;flex-shrink:0"></span>'
                        f'  {label} &nbsp;{score_chip} {reinf_chip}</div>'
                        f'<div class="cuser">You: {turn["user_message"]}</div>'
                        f'<div class="cbot">{turn["bot_response"]}</div>'
                        f'{corr_html}</div>',
                        unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

# Input
with st.form("cmp_form", clear_on_submit=True):
    cc1, cc2, cc3 = st.columns([4, 1, 1])
    with cc1:
        cmp_in = st.text_input("", "", placeholder="Send to all 3 conditions…", label_visibility="collapsed")
    with cc2:
        run_cmp = st.form_submit_button("Send All →", use_container_width=True)
    with cc3:
        reset_cmp = st.form_submit_button("Reset ↺", use_container_width=True)

if reset_cmp:
    for k in ["cmp_hist_base","cmp_hist_adap","cmp_hist_reinf"]:
        st.session_state[k] = []
    st.session_state.cmp_pending_correction = None
    st.rerun()

if run_cmp and cmp_in.strip():
    with st.spinner("Sending to all 3 conditions…"):
        new_correction = None
        for cond, hist_key, _ in [
            ("baseline",           "cmp_hist_base",  None),
            ("adaptive",           "cmp_hist_adap",  None),
            ("adaptive_reinforce", "cmp_hist_reinf", None),
        ]:
            hist    = st.session_state[hist_key]
            pending = st.session_state.cmp_pending_correction if cond == "adaptive_reinforce" else None
            t = run_resp(cmp_in, hist, persona, pending, cond)
            t["alignment_score"] = score_alignment(t["bot_response"], persona)
            if cond == "adaptive_reinforce":
                t = score_and_reinforce(t, threshold=threshold)
                new_correction = t.get("correction")
            else:
                t["correction"] = None
                t["reinforcement_applied"] = False
            st.session_state[hist_key].append(t)
        st.session_state.cmp_pending_correction = new_correction
    st.rerun()