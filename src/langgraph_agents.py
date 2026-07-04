"""
langgraph_agents.py
-------------------
A small, deterministic LangGraph workflow with three agents:

1. Data Analyst Agent   - explains the metrics in plain language.
2. Sports Insight Agent - adds short, coach-style observations.
3. QA Agent             - answers a user question using only the metrics.

Key rules:
- The LLM never sees the raw video, only the structured metrics JSON.
- Everything stays grounded; agents avoid unsupported claims.
- If there is no OpenAI API key (or a call fails), the app falls back to
  clear, metric-based text so it never crashes.

We use OpenAI's gpt-4o-mini via the official client, wired together with
LangGraph's StateGraph. Two tiny graphs are compiled: one for the report
(Analyst -> Insight) and one for interactive Q&A, so asking a follow-up
question does not re-run the whole report.
"""

from __future__ import annotations

import json
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from . import prompts
from .utils import get_openai_client, get_openai_model


def _movement_level_phrase(level: str) -> str:
    level = str(level or "unknown").lower()
    if level == "medium":
        return "moderate visible movement among tracked field segments"
    if level == "high":
        return "high visible pixel movement among tracked field segments"
    if level == "low":
        return "low visible pixel movement among tracked field segments"
    return "uncertain visible movement among tracked field segments"


class AnalysisState(TypedDict, total=False):
    metrics: dict
    question: str
    analyst_summary: str
    insight_text: str
    qa_answer: str
    llm_used: bool


def _metrics_json(metrics: dict) -> str:
    return json.dumps(metrics, indent=2)


def _call_llm(system: str, user: str) -> Optional[str]:
    """Call gpt-4o-mini. Returns None if there is no client or the call fails."""
    client = get_openai_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=get_openai_model(),
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None


# --- Deterministic fallbacks (used when no API key / call fails) --------
def _fallback_analyst(m: dict) -> str:
    movement_phrase = _movement_level_phrase(m.get("activity_level", "unknown"))
    return (
        f"Based on the tracking data, the clip is about "
        f"{m.get('video_duration_seconds', 0)} seconds long. The tracker created "
        f"{m.get('raw_track_ids_created', 0)} raw IDs and "
        f"{m.get('distinct_stable_tracks', 0)} distinct stable track segments, but "
        f"both over-count real players (they include brief detections, people in "
        f"view, and the same player picking up new IDs after occlusions). A more "
        f"realistic figure is about {m.get('avg_players_on_screen', 0)} field "
        f"detections visible on screen on average (peak "
        f"{m.get('peak_players_on_screen', 0)}). "
        f"{m.get('most_active_track_segment', 'One track segment')} showed the most "
        f"pixel movement, and the {str(m.get('highest_activity_zone', 'middle')).lower()} "
        f"third of the frame had the highest movement. Overall, this suggests "
        f"{movement_phrase}."
    )


def _fallback_insight(m: dict) -> str:
    zone = str(m.get("highest_activity_zone", "middle")).lower()
    return (
        f"- Movement appears higher on the {zone} side of the frame, which may "
        f"suggest more movement or positioning activity there during this clip.\n"
        f"- {m.get('most_active_track_segment', 'The top track segment')} covered the "
        f"most pixel distance, which may indicate higher movement for that segment.\n"
        f"- The busiest stretch appears to be around "
        f"{m.get('most_active_time_segment', 'N/A')}.\n"
        f"- Note: about {m.get('avg_players_on_screen', 0)} field detections are "
        f"visible on screen on average (peak {m.get('peak_players_on_screen', 0)}); "
        f"raw IDs and distinct track segments over-count. These are "
        f"pixel-movement observations only - the "
        f"system does not track the ball or teams."
    )


def _fallback_qa(m: dict, question: str) -> str:
    q = (question or "").lower().strip()
    if not q:
        return "Ask a question about the clip, e.g. 'Which side had the most movement?'"

    def has(*words: str) -> bool:
        return any(w in q for w in words)

    if has("zone", "side", "left", "right", "middle", "wing"):
        return (
            f"Based on the tracking data, the "
            f"{m.get('highest_activity_zone', 'middle')} third of the frame had "
            f"the most activity."
        )
    if has("how many", "number of player", "total player", "tracked", "count",
           "stable", "raw"):
        return (
            f"Based on the tracking data, about {m.get('avg_players_on_screen', 0)} "
            f"field detections are visible on screen on average (peak "
            f"{m.get('peak_players_on_screen', 0)}). The tracker created "
            f"{m.get('raw_track_ids_created', 0)} raw IDs and "
            f"{m.get('distinct_stable_tracks', 0)} distinct stable tracks, but both "
            f"over-count real players (people in view and ID switches after occlusion), "
            f"so they are not a headcount."
        )
    if has("when", "time", "moment", "segment", "period"):
        return (
            f"Based on the tracking data, the most active segment was "
            f"{m.get('most_active_time_segment', 'N/A')}."
        )
    if has("activity level", "high activity", "low activity", "intense", "busy"):
        return (
            f"Based on the tracking data, the clip shows "
            f"{_movement_level_phrase(m.get('activity_level', 'unknown'))}."
        )
    if has("move", "moved", "active segment", "most active", "run", "cover"):
        return (
            f"Based on the tracking data, "
            f"{m.get('most_active_track_segment', 'one track segment')} moved the most "
            f"by total pixel movement (a single player can span several track segments)."
        )
    return (
        "The current MVP does not measure that yet. It can answer questions about "
        "movement, zone activity, field detections visible on screen, the most "
        "active time segment, and the overall movement level. It does not track "
        "the ball or teams."
    )


# --- LangGraph nodes ----------------------------------------------------
def data_analyst_node(state: AnalysisState) -> dict:
    m = state.get("metrics", {})
    text = _call_llm(
        prompts.SYSTEM_GROUNDING,
        prompts.DATA_ANALYST_PROMPT.format(metrics_json=_metrics_json(m)),
    )
    used = text is not None
    return {"analyst_summary": text or _fallback_analyst(m), "llm_used": used}


def sports_insight_node(state: AnalysisState) -> dict:
    m = state.get("metrics", {})
    text = _call_llm(
        prompts.SYSTEM_GROUNDING,
        prompts.SPORTS_INSIGHT_PROMPT.format(
            analyst_summary=state.get("analyst_summary", ""),
            metrics_json=_metrics_json(m),
        ),
    )
    return {"insight_text": text or _fallback_insight(m)}


def qa_node(state: AnalysisState) -> dict:
    m = state.get("metrics", {})
    question = state.get("question", "")
    if not (question or "").strip():
        return {"qa_answer": ""}
    text = _call_llm(
        prompts.SYSTEM_GROUNDING,
        prompts.QA_PROMPT.format(metrics_json=_metrics_json(m), question=question),
    )
    return {"qa_answer": text or _fallback_qa(m, question)}


# --- Compile the two small graphs once ---------------------------------
def _build_report_graph():
    g = StateGraph(AnalysisState)
    g.add_node("data_analyst", data_analyst_node)
    g.add_node("sports_insight", sports_insight_node)
    g.add_edge(START, "data_analyst")
    g.add_edge("data_analyst", "sports_insight")
    g.add_edge("sports_insight", END)
    return g.compile()


def _build_qa_graph():
    g = StateGraph(AnalysisState)
    g.add_node("qa_agent", qa_node)
    g.add_edge(START, "qa_agent")
    g.add_edge("qa_agent", END)
    return g.compile()


_REPORT_GRAPH = _build_report_graph()
_QA_GRAPH = _build_qa_graph()


def _format_report(analyst: str, insight: str, llm_used: bool) -> str:
    header = "## AI Match Report"
    if not llm_used:
        header += (
            "  \n_(offline mode: no OpenAI API key found - showing a grounded, "
            "rule-based summary)_"
        )
    return (
        f"{header}\n\n"
        f"**Data Analyst**\n\n{analyst}\n\n"
        f"**Sports Insight**\n\n{insight}\n"
    )


# --- Public API used by app.py -----------------------------------------
def generate_report(metrics: dict) -> dict:
    """Run Analyst -> Insight and return a combined Markdown report."""
    state = _REPORT_GRAPH.invoke({"metrics": metrics, "question": ""})
    analyst = state.get("analyst_summary", "")
    insight = state.get("insight_text", "")
    llm_used = bool(state.get("llm_used", False))
    return {
        "report": _format_report(analyst, insight, llm_used),
        "analyst": analyst,
        "insight": insight,
        "llm_used": llm_used,
    }


def answer_question(metrics: dict, question: str) -> str:
    """Answer a single user question using only the metrics."""
    state = _QA_GRAPH.invoke({"metrics": metrics, "question": question})
    return state.get("qa_answer") or _fallback_qa(metrics, question)
