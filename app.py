"""
app.py  —  Persona-Adaptive Dialogue System
Run:  streamlit run app.py
 
New framing:
- User selects a persona at the start
- System maintains that persona against GPT's natural drift
- Baseline = weak prompt + high temperature (drifts visibly)
- Adaptive = strong persona prompt
- Adaptive + Reinforce = strong prompt + corrective loop
"""
 
import streamlit as st
import math, random
 
# ── Toggle ────────────────────────────────────────────────────────────────────
MOCK_MODE = False
from tone_detector   import detect_tone
from preset_matcher  import match_preset, smooth_persona
from dialogue_engine import get_bot_response
 
from persona_schema   import Persona, PRESETS
from alignment_scorer import score_alignment
from reinforcement    import score_and_reinforce
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PersonaBot", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family:'Inter',sans-serif !important; font-size:16px !important; }
h1 { font-family:'Syne',sans-serif !important; font-size:2.6rem !important; font-weight:800 !important; letter-spacing:-0.03em !important; margin-bottom:0.2rem !important; }
h3 { font-family:'Syne',sans-serif !important; font-size:1.05rem !important; font-weight:700 !important; margin:1.4rem 0 0.6rem !important; }
.bubble-user { background:#1a2035; border-left:3px solid #4f8ef7; padding:14px 18px; border-radius:8px; margin:10px 0 4px; font-size:1rem; line-height:1.7; color:#dce3f0; }
.bubble-bot  { background:#111825; border-left:3px solid #3dd68c; padding:14px 18px; border-radius:8px; margin:4px 0 10px; font-size:1rem; line-height:1.7; color:#dce3f0; }
.bubble-label { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.lbl-you { color:#4f8ef7; } .lbl-bot { color:#3dd68c; }
.chip { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; padding:2px 9px; border-radius:4px; font-weight:500; }
.chip-blue  { background:#0d1e3a; color:#4f8ef7; border:1px solid #1e3a6a; }
.chip-green { background:#0a2018; color:#3dd68c; border:1px solid #1a4030; }
.chip-amber { background:#1e1400; color:#f0a030; border:1px solid #3a2800; }
.chip-red   { background:#1e0a0a; color:#e05555; border:1px solid #3a1010; }
.chip-grey  { background:#151a25; color:#7a88aa; border:1px solid #2a3347; }
.corr-box { background:#180808; border-left:3px solid #e05555; padding:10px 14px; border-radius:6px; margin-top:8px; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; color:#cc5555; line-height:1.5; }
.trow  { display:flex; align-items:center; gap:10px; margin:6px 0; }
.tname { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#8898bb; width:82px; flex-shrink:0; }
.tbar  { flex:1; height:5px; background:#1c2a40; border-radius:3px; overflow:hidden; }
.tfill { height:100%; background:#4f8ef7; border-radius:3px; }
.tval  { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#4a5570; width:30px; text-align:right; }
.scard { background:#0e1525; border:1px solid #1c2a40; border-radius:8px; padding:16px 20px; margin:6px 0; }
.sval  { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; line-height:1; }
.slbl  { font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#5a6888; letter-spacing:0.08em; text-transform:uppercase; margin-top:4px; }
.ccard { background:#0e1525; border:1px solid #1c2a40; border-radius:8px; padding:18px 20px; height:100%; }
.clabel { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; color:#5a6888; margin-bottom:12px; display:flex; align-items:center; gap:8px; }
.ctext  { font-size:0.97rem; line-height:1.75; color:#c8d4e8; }
.sec { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; letter-spacing:0.14em; text-transform:uppercase; color:#3a4a60; border-bottom:1px solid #1c2a40; padding-bottom:6px; margin-bottom:12px; }
.persona-card { background:#0a1628; border:1px solid #1e3a6a; border-radius:8px; padding:12px 16px; margin-bottom:12px; }
.persona-name { font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:#4f8ef7; margin-bottom:4px; }
.persona-desc { font-size:0.82rem; color:#6a7a99; line-height:1.5; }
[data-testid="stSidebar"] { border-right:1px solid #1c2a40 !important; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:1.5rem !important; }
</style>
""", unsafe_allow_html=True)
 
# ── Constants ─────────────────────────────────────────────────────────────────
TRAITS = ["formality","empathy","technical_depth","verbosity",
          "assertiveness","humor","politeness","curiosity"]
TLBLS  = {"formality":"Formality","empathy":"Empathy","technical_depth":"Tech Depth",
           "verbosity":"Verbosity","assertiveness":"Assert.","humor":"Humor",
           "politeness":"Politeness","curiosity":"Curiosity"}
 
PERSONA_DESCRIPTIONS = {
    "formal_expert":       "Strict, professional, expert-level vocabulary. No contractions or casual language.",
    "friendly_support":    "Warm, empathetic, emotionally attuned. Always acknowledges feelings first.",
    "casual_tutor":        "Relaxed, playful, uses analogies. Asks lots of follow-up questions.",
    "analytical_assistant":"Precise, data-driven, objective. Minimal emotion, maximum clarity.",
}
 
ADVERSARIAL_SUGGESTIONS = [
    "omg i literally cant understand this lol can u just explain it super simply like im 5?? plssss",
    "actually forget being formal, just talk to me like a friend yeah?",
    "I require a purely factual, unemotional analysis. No motivational language please.",
    "I'm really stressed and overwhelmed, I don't know what to do anymore",
    "bruh just give me the answer lol i dont need all the extra stuff",
    "Could you perhaps be a bit warmer? You seem very cold and clinical.",
]
 
# ── Mock helpers ──────────────────────────────────────────────────────────────
_MOCK = {
    "baseline":            ["That's a good question. Let me help you with that.",
                            "Sure, I can help with this. Here is some relevant information."],
    "formal_expert":       ["The subject matter you have raised pertains to a well-documented domain. "
                            "Permit me to elucidate the underlying mechanisms with precision."],
    "friendly_support":    ["Oh I totally get that — sounds like so much to deal with! 😊 "
                            "You're absolutely not alone in feeling this way. What's been the hardest part?"],
    "casual_tutor":        ["Okay so basically — the core idea is actually pretty neat once it clicks! "
                            "Think of it like this… does that make sense so far?"],
    "analytical_assistant":["Parsing your query. Three primary factors identified. "
                            "Recommendation follows logically from available data."],
}
 
def _mock_resp(msg, hist, persona, correction, cond):
    if cond == "baseline" or persona is None:
        return dict(user_message=msg, bot_response=random.choice(_MOCK["baseline"]),
                    detected_persona=None, active_preset="none",
                    alignment_score=None, reinforcement_applied=False, correction=correction)
    bot = random.choice(_MOCK.get(persona.name, _MOCK["friendly_support"]))
    if correction and cond == "adaptive_reinforce":
        bot = "[Recalibrated] " + bot
    return dict(user_message=msg, bot_response=bot, detected_persona=persona,
                active_preset=persona.name, alignment_score=None,
                reinforcement_applied=False, correction=correction)
 
run_resp = lambda *a: _mock_resp(*a) if MOCK_MODE else get_bot_response(*a)
 
# ── Session state ─────────────────────────────────────────────────────────────
for k, v in dict(
    hist_base=[], hist_adap=[], hist_reinf=[],
    selected_preset="formal_expert",
    pending_correction=None,
    turn_count=0
).items():
    if k not in st.session_state:
        st.session_state[k] = v
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def sc(v): return "chip-green" if v >= 0.72 else ("chip-amber" if v >= 0.52 else "chip-red")
def sc_color(v): return "#3dd68c" if v >= 0.72 else ("#f0a030" if v >= 0.52 else "#e05555")
 
def trait_bars(persona):
    html = ""
    for t in TRAITS:
        val = getattr(persona, t); pct = int(val * 100)
        html += f'<div class="trow"><span class="tname">{TLBLS[t]}</span><div class="tbar"><div class="tfill" style="width:{pct}%"></div></div><span class="tval">{val:.2f}</span></div>'
    return html
 
def radar(scores, persona, sz=240):
    n = len(TRAITS); cx = cy = sz // 2; r = sz // 2 - 38
    short = ["Form","Emp","Tech","Verb","Asrt","Hmr","Pol","Cur"]
    def pt(i, v):
        a = math.pi / 2 + 2 * math.pi * i / n
        return cx + r * v * math.cos(a), cy - r * v * math.sin(a)
    grid  = "".join(f'<polygon points="{" ".join(f"{pt(i,lv)[0]:.1f},{pt(i,lv)[1]:.1f}" for i in range(n))}" fill="none" stroke="#1c2a40" stroke-width="1"/>' for lv in [.25,.5,.75,1.])
    axes  = "".join(f'<line x1="{cx}" y1="{cy}" x2="{pt(i,1)[0]:.1f}" y2="{pt(i,1)[1]:.1f}" stroke="#1c2a40" stroke-width="1"/>' for i in range(n))
    tp    = " ".join(f"{pt(i,getattr(persona,TRAITS[i]))[0]:.1f},{pt(i,getattr(persona,TRAITS[i]))[1]:.1f}" for i in range(n))
    tgt   = f'<polygon points="{tp}" fill="#4f8ef710" stroke="#4f8ef7" stroke-width="1.5" stroke-dasharray="4,3"/>'
    sv    = [scores.get(t, .5) for t in TRAITS]
    sp    = " ".join(f"{pt(i,sv[i])[0]:.1f},{pt(i,sv[i])[1]:.1f}" for i in range(n))
    scr   = f'<polygon points="{sp}" fill="#3dd68c18" stroke="#3dd68c" stroke-width="2"/>'
    dots  = "".join(f'<circle cx="{pt(i,sv[i])[0]:.1f}" cy="{pt(i,sv[i])[1]:.1f}" r="3.5" fill="#3dd68c"/>' for i in range(n))
    lbls  = ""
    for i, lbl in enumerate(short):
        lx, ly = pt(i, 1.36); anc = "middle"
        if lx < cx - 8: anc = "end"
        elif lx > cx + 8: anc = "start"
        lbls += f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anc}" font-size="10" fill="#6a7a99" font-family="IBM Plex Mono,monospace">{lbl}</text>'
    leg = (f'<rect x="8" y="{sz-22}" width="12" height="2" fill="#4f8ef7" opacity=".7"/>'
           f'<text x="24" y="{sz-17}" font-size="9" fill="#5a6888" font-family="IBM Plex Mono,monospace">target persona</text>'
           f'<rect x="8" y="{sz-13}" width="12" height="2" fill="#3dd68c"/>'
           f'<text x="24" y="{sz-8}" font-size="9" fill="#5a6888" font-family="IBM Plex Mono,monospace">response scores</text>')
    return (f'<svg width="{sz}" height="{sz}" xmlns="http://www.w3.org/2000/svg" style="border-radius:10px;display:block">'
            f'{grid}{axes}{tgt}{scr}{dots}{lbls}{leg}</svg>')
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
 
    # ── Persona selector — the new key UI element ─────────────────────────────
    st.markdown('<div class="sec">◈ &nbsp;Select Persona</div>', unsafe_allow_html=True)
    selected = st.selectbox("",
        options=list(PRESETS.keys()),
        format_func=lambda x: {
            "formal_expert":       "🎓  Formal Expert",
            "friendly_support":    "🤗  Friendly Support",
            "casual_tutor":        "📚  Casual Tutor",
            "analytical_assistant":"🔬  Analytical Assistant",
        }[x],
        label_visibility="collapsed",
        key="selected_preset"
    )
 
    # Show persona description + trait bars
    persona = PRESETS[selected]
    st.markdown(
        f'<div class="persona-card">'
        f'<div class="persona-desc">{PERSONA_DESCRIPTIONS[selected]}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(trait_bars(persona), unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── Condition ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Condition</div>', unsafe_allow_html=True)
    condition = st.radio("",
        ["baseline", "adaptive", "adaptive_reinforce"],
        format_func=lambda x: {
            "baseline":           "① Baseline (no control)",
            "adaptive":           "② Persona-Controlled",
            "adaptive_reinforce": "③ Persona + Reinforce",
        }[x],
        label_visibility="collapsed"
    )
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── Threshold ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Reinforcement Threshold</div>', unsafe_allow_html=True)
    threshold = st.slider("", 0.3, 0.9, 0.65, 0.05, label_visibility="collapsed")
    st.caption(f"Fires corrective prompt when alignment < {threshold:.2f}")
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── Adversarial suggestions ───────────────────────────────────────────────
    st.markdown('<div class="sec">💡 Try These (Adversarial)</div>', unsafe_allow_html=True)
    st.caption("Messages designed to pull GPT away from the persona — best for showing the difference between conditions.")
    for sug in ADVERSARIAL_SUGGESTIONS[:3]:
        st.caption(f"› {sug[:60]}…")
 
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↺  Reset conversation", use_container_width=True):
        for k in ["hist_base", "hist_adap", "hist_reinf"]:
            st.session_state[k] = []
        st.session_state.pending_correction = None
        st.session_state.turn_count = 0
        st.rerun()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
hkey    = {"baseline":"hist_base","adaptive":"hist_adap","adaptive_reinforce":"hist_reinf"}[condition]
history = st.session_state[hkey]
persona = PRESETS[st.session_state.selected_preset]
 
st.markdown("# 🧠 Persona-Adaptive Dialogue")
st.caption("NLP Course Project  ·  Live Demo  ·  April 2026")
 
if MOCK_MODE:
    st.warning("⚠ MOCK MODE — set `MOCK_MODE = False` to use the live engine", icon="⚠️")
 
# Research framing banner
st.info(
    f"**Research question:** Can a structured persona enforcement system make GPT maintain "
    f"a consistent personality better than naive prompting alone? "
    f"  \n**Active persona:** `{st.session_state.selected_preset}` — "
    f"try sending adversarial messages (see sidebar) to see how each condition responds.",
    icon="🔬"
)
 
st.divider()
 
chat_col, panel_col = st.columns([3, 1], gap="large")
 
# ── Chat ──────────────────────────────────────────────────────────────────────
with chat_col:
    clabels = {
        "baseline":           "① Baseline — no persona control",
        "adaptive":           "② Persona-Controlled",
        "adaptive_reinforce": "③ Persona + Reinforcement Loop",
    }
    st.markdown(f'<div class="sec">Conversation &nbsp;·&nbsp; {clabels[condition]}</div>', unsafe_allow_html=True)
 
    for turn in history:
        st.markdown(f'<div class="bubble-user"><div class="bubble-label lbl-you">You</div>{turn["user_message"]}</div>', unsafe_allow_html=True)
 
        preset  = turn.get("active_preset", "—")
        scores  = turn.get("alignment_score") or {}
        overall = scores.get("overall")
 
        preset_chip = f'<span class="chip chip-blue">⬡ {preset}</span>' if preset != "none" \
                      else '<span class="chip chip-grey">no persona</span>'
        score_chip  = f'<span class="chip {sc(overall)}">{overall:.2f}</span>' if overall is not None else ""
        reinf_chip  = '<span class="chip chip-red">🔧 corrected</span>' if turn.get("reinforcement_applied") else ""
        corr_html   = f'<div class="corr-box">⚡ Correction injected → {turn["correction"]}</div>' if turn.get("correction") else ""
 
        st.markdown(
            f'<div class="bubble-bot">'
            f'<div class="bubble-label lbl-bot">{preset_chip} {score_chip} {reinf_chip}</div>'
            f'{turn["bot_response"]}{corr_html}</div>',
            unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            user_input = st.text_input("", "", placeholder="Type your message… or try an adversarial message from the sidebar", label_visibility="collapsed")
        with c2:
            submitted = st.form_submit_button("Send →", use_container_width=True)
 
    if submitted and user_input.strip():
        with st.spinner("Thinking…"):
            # Persona is now user-selected, not detected
            active_persona = persona if condition != "baseline" else None
            pending = st.session_state.pending_correction if condition == "adaptive_reinforce" else None
 
            turn = run_resp(user_input, history, active_persona, pending, condition)
 
            if condition in ("adaptive", "adaptive_reinforce"):
                turn = score_and_reinforce(turn, threshold=threshold)
                st.session_state.pending_correction = turn.get("correction") if condition == "adaptive_reinforce" else None
            else:
                st.session_state.pending_correction = None
 
            st.session_state[hkey].append(turn)
            st.session_state.turn_count += 1
        st.rerun()
 
 
# ── Right panel ───────────────────────────────────────────────────────────────
with panel_col:
    st.markdown('<div class="sec">Persona Radar</div>', unsafe_allow_html=True)
    if history:
        last   = history[-1]
        scores = last.get("alignment_score") or {}
        if scores and last.get("detected_persona"):
            st.markdown(radar(scores, last["detected_persona"]), unsafe_allow_html=True)
        else:
            st.caption("No alignment data yet — switch to Adaptive condition.")
    else:
        st.caption("Send a message to see the radar.")
 
    if history:
        last   = history[-1]
        scores = last.get("alignment_score") or {}
        if scores:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sec">Last Turn Scores</div>', unsafe_allow_html=True)
            rows = ""
            for t in ["formality","empathy","technical_depth","verbosity","politeness","curiosity"]:
                v = scores.get(t, 0.)
                rows += f'<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid #141e2e"><span style="font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#7a8aaa">{TLBLS[t]}</span><span class="chip {sc(v)}" style="font-size:0.75rem">{v:.2f}</span></div>'
            ov = scores.get("overall", 0.)
            rows += f'<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0 0"><span style="font-family:IBM Plex Mono,monospace;font-size:0.82rem;color:#c8d4e8;font-weight:600">Overall</span><span class="chip {sc(ov)}" style="font-size:0.82rem;padding:3px 10px">{ov:.2f}</span></div>'
            st.markdown(rows, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">Session Stats</div>', unsafe_allow_html=True)
    reinf = sum(1 for t in history if t.get("reinforcement_applied"))
    avgs  = [t["alignment_score"]["overall"] for t in history if isinstance(t.get("alignment_score"), dict)]
    avg   = sum(avgs) / len(avgs) if avgs else 0.
 
    st.markdown(f"""
    <div class="scard"><div class="sval">{len(history)}</div><div class="slbl">Turns</div></div>
    <div class="scard"><div class="sval" style="color:#e05555">{reinf}</div><div class="slbl">Corrections Fired</div></div>
    <div class="scard"><div class="sval" style="color:{sc_color(avg)}">{avg:.2f}</div><div class="slbl">Avg Alignment</div></div>
    """, unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SIDE-BY-SIDE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### 🔬 Side-by-Side Comparison")
st.caption(
    f"Same message, same persona (`{st.session_state.selected_preset}`), three conditions. "
    "Best demo moment — use an adversarial message for maximum contrast."
)
 
with st.form("cmp_form"):
    cc1, cc2 = st.columns([4, 1])
    with cc1:
        cmp_in = st.text_input("", "",
            placeholder="e.g.  omg just explain it super simply lol i dont get it at all",
            label_visibility="collapsed")
    with cc2:
        run_cmp = st.form_submit_button("Run All →", use_container_width=True)
 
if run_cmp and cmp_in.strip():
    cols = st.columns(3, gap="medium")
    configs = [
        ("baseline",           "① Baseline",            "#4a5570", None),
        ("adaptive",           "② Persona-Controlled",  "#4f8ef7", persona),
        ("adaptive_reinforce", "③ Persona + Reinforce", "#3dd68c", persona),
    ]
    for col, (cond, label, color, p) in zip(cols, configs):
        t = run_resp(cmp_in, [], p, None, cond)
        if cond != "baseline":
            t = score_and_reinforce(t, threshold=threshold)
        ov  = (t.get("alignment_score") or {}).get("overall")
        sc_chip = f'<span class="chip {sc(ov)}">{ov:.2f}</span>' if ov else ""
        with col:
            st.markdown(f"""
            <div class="ccard">
              <div class="clabel">
                <span style="width:9px;height:9px;border-radius:50%;background:{color};display:inline-block;flex-shrink:0"></span>
                {label} &nbsp;{sc_chip}
              </div>
              <div class="ctext">{t["bot_response"]}</div>
            </div>""", unsafe_allow_html=True)