import os
import json
import random
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st

def _print(string):
    print(string, flush=True)

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="3-Agent Self-Discovery Chat",
    page_icon="🧭",
    layout="centered",
)

# -----------------------------
# Utilities
# -----------------------------
AGENT_NAMES = {
    "intro": "Introduction Agent",
    "scenario": "Scenario Agent",
    "analyst": "Analyst Agent",
}

DEFAULT_MODEL = "gpt-5-nano"

INTAKE_SYSTEM = (
    "You are a warm, curious intake coach. Your goal is to get to know
the user — "
    "their background, mood, goals, interests, and what they hope to
discover about themselves. "
    "Ask one open question at a time, phrased in simple,
conversational language. "
    "Acknowledge their responses briefly before moving on. "
    "Make it feel light and human, not like a form. "
    "End once you have enough context to understand who they are and
what they want from this experience. "
    "Keep replies under 120 words."
)

INNER_MIRROR_SYSTEM = (
    "You are a playful, reflective guide helping the user explore both
personality and purpose. "
    "Create a sequence of 25–30 questions mixing quick multiple-choice
personality prompts and short reflective value questions. "
    "Use real-life, relatable situations that feel fun, authentic, and
emotionally safe. "
    "Keep questions short and avoid jargon. "
    "For personality questions: offer 4 options that reflect different
tendencies (no right or wrong). "
    "For values questions: invite reflection in one or two sentences. "
    "Your tone is curious, positive, and slightly introspective — like
a friend guiding a self-discovery game. "
    "Keep each message under 120 words."
)

ARCHETYPE_SYSTEM = (
    "You are an insightful analyst and storyteller. "
    "Read Conversation A (intake) and Conversation B (inner mirror). "
    "Synthesize patterns into a concise personality report with an
animal archetype that captures the user’s essence. "
    "Describe their energy, motivations, strengths, and growth edges
in a warm, empowering way. "
    "Suggest which professional roles, work styles, or environments
they are most naturally aligned with. "
    "Include sections: Archetype, Description (3–4 sentences),
Strengths (3–5), Growth Areas (2–3), "
    "Ideal Environment, Best-Fit Roles (3–5 examples such as 'Creative
Strategist' or 'Community Builder'), "
    "and Mantra. "
    "Use vivid, natural language that feels personal and uplifting. "
    "Keep the full report under 300 words."
)

INTRO_SYSTEM = INTAKE_SYSTEM
SCENARIO_SYSTEM = INNER_MIRROR_SYSTEM
ANALYST_SYSTEM = ARCHETYPE_SYSTEM




SCENARIOS = [
    "You’ve been offered two projects: (A) high-visibility, tight deadline; (B) low-pressure, deep craftsmanship. Which do you choose and why?",
    "Your team disagrees with your proposed approach. How do you respond in the next meeting?",
    "You have a free Friday with no obligations. How do you spend it to feel fulfilled?",
    "A close colleague gives you constructive criticism that stings. What’s your first reaction, and what do you do?",
    "You can learn one new skill in 30 days. What is it and what small plan do you make to practice daily?",
]

# -----------------------------
# LLM Backends
# -----------------------------

def use_llm() -> bool:
    key = st.session_state.get("api_key") or os.getenv("OPENAI_API_KEY")
    return bool(key)



def _openai_chat(messages: List[Dict[str, str]], model: str, temperature: float = 0.7) -> str:
    """Call OpenAI Chat Completions using the modern openai>=1.0.0 SDK."""
    api_key = st.session_state.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OpenAI API key.")

    try:
        from openai import OpenAI  # modern SDK
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")


def offline_stub(agent: str, user_text: str) -> str:
    """Deterministic stub to make the app usable without an API key."""
    if agent == "intro":
        return (
            "[offline demo] Thanks! I heard: '" + user_text +
            "'. What matters to you lately—energy, focus, relationships, or growth?"
        )
    if agent == "scenario":
        return (
            "[offline demo] Interesting! Given that, what would be your first small step and why?"
        )
    if agent == "analyst":
        return (
            "[offline demo]\n\nKey Themes: curiosity, steady growth, care for quality.\n"
            "Strengths: reflection, empathy.\nGrowth Areas: prioritization, saying no.\n"
            "Suggested Experiments: 1) 20-min daily planning, 2) feedback sandwich once/week, 3) 2-hour focus block."
        )
    return "[offline demo]"


def agent_reply(agent: str, messages: List[Dict[str, str]], model: str, temperature: float) -> str:
    if use_llm():
        try:
            _print("Trying OpenAI chat")
            return _openai_chat(messages, model=model, temperature=temperature)
        except Exception as e:
            _print(f"LLM call failed: {e}. Falling back to offline demo mode.")
            st.warning(f"LLM call failed: {e}. Falling back to offline demo mode.")
            return offline_stub(agent, messages[-1]["content"]) if messages else "[offline demo]"
    else:
        return offline_stub(agent, messages[-1]["content"]) if messages else "[offline demo]"

# -----------------------------
# Session State
# -----------------------------

def init_state():
    st.session_state.setdefault("phase", 1)  # 1=intro, 2=scenario, 3=analyst
    st.session_state.setdefault("api_key", "")
    st.session_state.setdefault("model", DEFAULT_MODEL)
    st.session_state.setdefault("temperature", 0.7)

    # Chats store minimal role/content pairs for replay
    st.session_state.setdefault("intro_chat", [])  # List[Dict[str,str]]
    st.session_state.setdefault("scenario_chat", [])

    # Scenario prompt derived/selected at phase 2
    st.session_state.setdefault("scenario_prompt", None)

    # Analyst output text
    st.session_state.setdefault("analysis", None)

    # System messages
    st.session_state.setdefault("intro_system", INTRO_SYSTEM)
    st.session_state.setdefault("scenario_system", SCENARIO_SYSTEM)
    st.session_state.setdefault("analyst_system", ANALYST_SYSTEM)


init_state()

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Settings")
st.session_state.api_key = st.sidebar.text_input(
    "OpenAI API Key (optional)",
    type="password",
    placeholder="sk-...",
    value=st.session_state.get("api_key", ""),
)

st.session_state.model = st.sidebar.selectbox(
    "Model",
    [DEFAULT_MODEL, "gpt-5-mini","gpt-5"],
    index=0,
)

if st.sidebar.button("🔄 Start Over"):
    for key in [
        "phase", "intro_chat", "scenario_chat", "scenario_prompt", "analysis"
    ]:
        st.session_state[key] = None if key in ("scenario_prompt", "analysis") else [] if "chat" in key else 1
    st.rerun()

# -----------------------------
# Header / Stepper
# -----------------------------
st.title("🧭 3-Agent Self-Discovery Chat")

phase = st.session_state.phase
step_cols = st.columns(3)
steps = ["1) Meet", "2) Explore", "3) Analyze"]
for i, col in enumerate(step_cols, start=1):
    with col:
        label = steps[i-1]
        if phase == i:
            st.markdown(f"**➡️ {label}**")
        elif phase > i:
            st.markdown(f"✅ {label}")
        else:
            st.markdown(label)

st.markdown("---")

# -----------------------------
# Phase 1: Introduction Agent
# -----------------------------
if phase == 1:
    st.subheader(f"{AGENT_NAMES['intro']}")

    # Seed first message
    if not st.session_state.intro_chat:
        opening = "Hi! I’d love to get to know you. What would you like to understand or improve about yourself right now?"
        st.session_state.intro_chat.append({"role": "assistant", "content": opening})

    # Render chat
    for m in st.session_state.intro_chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # User input
    user_msg = st.chat_input("Share something about you…")
    if user_msg:
        st.session_state.intro_chat.append({"role": "user", "content": user_msg})

        # Build messages with system
        messages = ([{"role": "system", "content": st.session_state.intro_system}] +
                    st.session_state.intro_chat)
        reply = agent_reply("intro", messages, model=st.session_state.model, temperature=st.session_state.temperature)
        st.session_state.intro_chat.append({"role": "assistant", "content": reply})
        st.rerun()

    st.caption("Tip: after a couple of exchanges, click **Next** to explore a scenario.")
    if st.button("Next → Scenario"):
        st.session_state.phase = 2
        st.rerun()

# -----------------------------
# Phase 2: Scenario Agent
# -----------------------------
elif phase == 2:
    st.subheader(f"{AGENT_NAMES['scenario']}")

    # Ensure a scenario prompt exists
    if st.session_state.scenario_prompt is None:
        # Try to draft a scenario seeded by the intro conversation
        intro_summary_request = (
            "Based on the prior intake chat, pick ONE short scenario that will best surface the user's preferences. "
            "Choose from themes like: decision trade-off, feedback response, time use, or learning plan. "
            "Return only the scenario statement, one or two sentences."
        )
        # Build messages
        seed_messages = [
            {"role": "system", "content": st.session_state.scenario_system},
            {"role": "user", "content": intro_summary_request},
            {"role": "assistant", "content": "(You have access to the intake chat below.)"},
            {"role": "user", "content": "Intake chat transcript:\n" + json.dumps(st.session_state.intro_chat, ensure_ascii=False)},
        ]
        try:
            scenario_text = agent_reply("scenario", seed_messages, model=st.session_state.model, temperature=0.6).strip()
        except Exception:
            scenario_text = None

        if not scenario_text or scenario_text.startswith("[offline demo]") or len(scenario_text) > 400:
            scenario_text = random.choice(SCENARIOS)

        st.session_state.scenario_prompt = scenario_text
        st.session_state.scenario_chat.append({"role": "assistant", "content": f"Scenario: {scenario_text}"})

    # Render chat
    for m in st.session_state.scenario_chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Input
    user_msg = st.chat_input("How would you respond in this scenario?")
    if user_msg:
        st.session_state.scenario_chat.append({"role": "user", "content": user_msg})

        messages = ([{"role": "system", "content": st.session_state.scenario_system}] +
                    st.session_state.scenario_chat)
        reply = agent_reply("scenario", messages, model=st.session_state.model, temperature=st.session_state.temperature)
        st.session_state.scenario_chat.append({"role": "assistant", "content": reply})
        st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔁 New Scenario"):
            st.session_state.scenario_prompt = None
            st.session_state.scenario_chat = []
            st.rerun()
    with c2:
        if st.button("Next → Analyst"):
            st.session_state.phase = 3
            st.rerun()

# -----------------------------
# Phase 3: Analyst Agent
# -----------------------------
else:
    st.subheader(f"{AGENT_NAMES['analyst']}")

    if st.session_state.analysis is None:
        # Compose analysis prompt
        analysis_user = (
            "Please analyze these two conversations and produce a concise report with sections: "
            "Key Themes, Strengths, Growth Areas, Suggested Experiments (3-5)."
        )
        analysis_messages = [
            {"role": "system", "content": st.session_state.analyst_system},
            {"role": "user", "content": analysis_user},
            {"role": "assistant", "content": "Conversation A (intake):"},
            {"role": "user", "content": json.dumps(st.session_state.intro_chat, ensure_ascii=False)},
            {"role": "assistant", "content": "Conversation B (scenario):"},
            {"role": "user", "content": json.dumps(st.session_state.scenario_chat, ensure_ascii=False)},
        ]

        analysis_text = agent_reply("analyst", analysis_messages, model=st.session_state.model, temperature=0.4)
        st.session_state.analysis = analysis_text

    # Show analysis
    st.markdown(st.session_state.analysis)

    # Download artifacts
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "intro_chat": st.session_state.intro_chat,
        "scenario_chat": st.session_state.scenario_chat,
        "scenario_prompt": st.session_state.scenario_prompt,
        "analysis": st.session_state.analysis,
        "model": st.session_state.model,
        "temperature": st.session_state.temperature,
    }
    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    st.download_button(
        "⬇️ Download JSON Report",
        data=report_json,
        file_name="three_agent_report.json",
        mime="application/json",
    )

    st.caption("You can go back to **Meet** or **Explore** from the sidebar -> Start Over.")
