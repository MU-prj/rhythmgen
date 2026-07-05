"""rhythmgen: generazione di ritmi con ascoltatore interno.

Design e decisioni in docs/design.md.
"""

from .generator import BernoulliGenerator
from .grid import build_grid, metric_weight
from .render import Event, events_to_json, render

__all__ = [
    "BernoulliGenerator",
    "Event",
    "build_grid",
    "events_to_json",
    "metric_weight",
    "render",
]
