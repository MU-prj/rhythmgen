"""rhythmgen: generazione di ritmi con ascoltatore interno.

Design e decisioni in docs/design.md.
"""

from .blind import BlindListener
from .control import (
    ClosedLoop,
    LoopResult,
    LowPass,
    PIController,
    SetpointCurve,
    TracePoint,
    trace_to_json,
)
from .generator import BernoulliGenerator
from .grid import build_grid, metric_weight
from .listener import (
    Level,
    OracleListener,
    ScoredEvent,
    SurpriseRMS,
    default_levels,
    leaky_rms,
    pulse,
)
from .render import Event, events_to_json, render
from .syncopation import lhl_syncopation, toussaint_complexity

__all__ = [
    "BernoulliGenerator",
    "BlindListener",
    "ClosedLoop",
    "Event",
    "Level",
    "LoopResult",
    "LowPass",
    "OracleListener",
    "PIController",
    "ScoredEvent",
    "SetpointCurve",
    "SurpriseRMS",
    "TracePoint",
    "build_grid",
    "default_levels",
    "events_to_json",
    "leaky_rms",
    "lhl_syncopation",
    "metric_weight",
    "pulse",
    "render",
    "toussaint_complexity",
    "trace_to_json",
]
