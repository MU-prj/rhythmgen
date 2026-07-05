"""Rendering temporale: posizioni razionali → secondi (design D8, D9).

Fase 1: conversione deterministica più un unico jitter gaussiano i.i.d.
(inapprendibile), default 0. Swing e groove sistematici sono fase 2.
"""

import json
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Optional


@dataclass(frozen=True)
class Event:
    onset_s: float
    position: Fraction  # posizione assoluta in beat, esatta (debug/analisi)
    velocity: float


def render(
    events: Iterable[tuple[Fraction, float]],
    bpm: float,
    sigma_t: float = 0.0,
    rng: Optional[random.Random] = None,
) -> list[Event]:
    """Converte (posizione in beat, velocity) in eventi temporali.

    ``sigma_t`` è la deviazione standard del jitter in secondi. Il jitter può
    riordinare onset vicinissimi: l'ascoltatore riceve comunque timestamp, non
    la griglia, quindi non è un errore (D2).
    """
    beat_s = 60.0 / bpm
    if sigma_t > 0 and rng is None:
        rng = random.Random()
    return [
        Event(
            float(pos) * beat_s + (rng.gauss(0.0, sigma_t) if sigma_t > 0 else 0.0),
            pos,
            vel,
        )
        for pos, vel in events
    ]


def events_to_json(events: Iterable[Event]) -> str:
    """Serializzazione JSON (D9): la posizione razionale diventa "num/den"."""
    return json.dumps(
        [
            {"onset_s": e.onset_s, "position": str(e.position), "velocity": e.velocity}
            for e in events
        ],
        indent=2,
    )
