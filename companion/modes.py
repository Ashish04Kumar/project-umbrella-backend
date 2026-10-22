"""How the user is talking to the companion, and which extras each way of talking switches on.

Plug and play: to turn voice cues on or off for a mode, change its constant here. Voice cues
(how the user sounded: pause before answering, pace, loudness, intonation, long pauses, trailing
off, fillers, cutting in) are measured and passed to the LLM only where `voice_cues=True`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InputMode:
    name: str
    voice_cues: bool  # measure how the user sounded and give the LLM a soft hint about it


TEXT = InputMode("text", voice_cues=False)                          # typed messages
PUSH_TO_TALK = InputMode("push_to_talk", voice_cues=False)          # 🎤: speech -> words only
PUSH_TO_TALK_CUES = InputMode("push_to_talk_cues", voice_cues=True)  # 🎙️✨: speech -> words + voice cues
LIVE = InputMode("live", voice_cues=True)                           # 📞: real-time voice over LiveKit

MODES = {mode.name: mode for mode in (TEXT, PUSH_TO_TALK, PUSH_TO_TALK_CUES, LIVE)}
