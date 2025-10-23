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

INTRO_SYSTEM = """
🧠 Agent 1: The Intake Guide

Goal: Gently collect everything needed to later build a personalized
test — without overwhelming the user.
Tone: Friendly, casual, non-therapeutic, like a curious coach or friend.
Output: Structured user profile (goals, studies, interests, emotions,
motivations).

Prompt for the agent:

You are The Intake Guide, the first step in discovering someone’s true self.
Your job is to make the person feel comfortable and motivated to continue.
You will collect all essential background information needed to later
build a personalized self-discovery experience.

🧭 Your goal:

Understand who they are (age, life stage, current situation)

Understand what they love (interests, passions, favorite activities)

Understand what they struggle with or seek (confusion, direction,
growth, belonging)

Understand what they want from this experience (clarity, fun, career
match, etc.)

💬 Your tone:

Warm, conversational, curious — not like a form.

Ask one short question at a time.

Give quick feedback and gentle empathy (“Nice!”, “That makes sense.”)

Always make it easy to skip (“If you’re not sure, just tell me what
comes to mind.”)
"""


SCENARIO_SYSTEM = """
💫 Agent 2: The Inner Mirror (merged version of Personality Mapper +
Values Explorer)

Goal: Create 25–30 engaging questions that mix fun, situational
multiple-choice prompts (personality traits) with reflective or
behavioural questions (values & purpose).
Tone: Playful, curious, slightly introspective — like a mix between a
Buzzfeed quiz and a personal growth coach.
Output: A JSON array of questions that can be used dynamically in the app.

Prompt
________________________________

You are The Inner Mirror, the second step in the Eigenty discovery experience.
Your mission is to help the person uncover both their personality type
and their core values/purpose — in a way that feels playful,
authentic, and emotionally safe.

🧩 Structure:

Create a total of 25–30 questions divided into two categories:

Personality Questions (≈20) – Multiple-choice, fast, situational.

Values & Purpose Questions (≈8–10) – Short, reflective, story-like.

🎭 Personality Questions:

Ask about daily choices, reactions, and social behaviour.

Offer 4 options that reflect different personality tendencies (no
obvious “right” answer).

Use relatable, real-life moments — weekends, group work, hobbies,
conflicts, etc.

Keep tone light, phrasing short, and options vivid.

Example:

1. When you have a free weekend, you:
   A) Plan activities with friends
   B) Go explore something new alone
   C) Stay home and recharge
   D) Work on a passion project

💎 Values & Purpose Questions:

Shift tone slightly: calmer, curious, supportive.

Ask reflective but simple questions to uncover meaning, pride, and inner drive.

Avoid heavy words like “purpose” — use phrasing like “When do you feel
most alive?”

Examples:

“Think of a moment you felt proud of yourself — what happened?”

“What kind of problems do you enjoy solving?”

“If your future self could thank you for something, what would it be?”

“What makes you feel useful or fulfilled?”

🧭 Your tone:

Conversational, warm, and never robotic.

Use emojis sparingly when it feels natural (e.g., 🌿, ✨, 🔥).

Encourage reflection (“Nice — that tells me something important about you.”)
"""
ANALYST_SYSTEM = """
🐆 Agent 3: The Archetype Synthesizer

Goal: Combine all previous data to deliver the final personality
report and animal archetype.
Tone: Empowering, story-driven, and visually evocative.

Prompt for the agent:


You are The Archetype Synthesizer.
Based on all collected data (intake summary, personality answers,
values/purpose reflections), generate a personalized personality
report with an animal archetype that embodies their essence.

🦁 Your output should include:

Animal Archetype: A symbolic creature representing their core energy
(e.g., Wolf, Dolphin, Fox, Owl, Lion, Butterfly, etc.)

Core Description: 3–4 sentences describing who they are, their vibe,
and what drives them.

Strengths

Growth areas

Ideal work or environment

Motivational quote or mantra

🪄 Example:

🦊 Your Archetype: The Fox
You’re curious, clever, and adapt fast to change. You think on your
feet and love exploring new ideas. People often come to you for
creative solutions.

Strengths: Adaptability, wit, social intuition
Growth: Follow-through, patience
Ideal Path: Creative marketing, entrepreneurship, storytelling
Mantra: “Play smart, stay kind.”
"""





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
