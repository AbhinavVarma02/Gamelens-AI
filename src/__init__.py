"""
GameLens AI source package.

Small, focused modules that together turn a short soccer clip into
structured tracking data, simple analytics, charts, and grounded
LLM insights:

- video_processor : OpenCV video read/write helpers
- detect_track    : YOLO detection + ByteTrack tracking
- analytics       : tracking CSV + data-science metrics
- visualizations  : Matplotlib charts
- prompts         : LLM prompt templates
- langgraph_agents: 3-agent LangGraph workflow (Analyst / Insight / QA)
- utils           : shared config and helpers
"""

__version__ = "1.0.0"
