"""Generatore a campo di Bernoulli parametrico (design D7)."""

import random
from fractions import Fraction
from typing import Iterable

from .grid import build_grid, metric_weight


def _tilt(w: float, syncopation: float) -> float:
    """Profilo inclinato w' = (1-σ)·w + σ·(1-w): a σ=1 il profilo si inverte."""
    return (1.0 - syncopation) * w + syncopation * (1.0 - w)


def _is_binary(pos: Fraction) -> bool:
    d = (pos % 1).denominator
    return d & (d - 1) == 0


class BernoulliGenerator:
    """Campionamento Bernoulli indipendente per posizione, senza memoria tra
    misure (D7). I parametri si passano a ogni chiamata: il controllore (M4)
    li modulerà nel tempo senza stato dentro il generatore.

    ``density`` è in eventi attesi per beat e può superare 1; la
    normalizzazione garantisce Σp = density·beats finché nessuna p satura a 1.
    """

    def __init__(self, beats: int, subdivisions: Iterable[int]):
        self.beats = beats
        self.grid = build_grid(beats, subdivisions)
        self._weights = {p: metric_weight(p, beats) for p in self.grid}

    def probabilities(
        self,
        *,
        density: float = 1.0,
        syncopation: float = 0.0,
        tuplet_mix: float = 0.0,
    ) -> dict[Fraction, float]:
        """p(pos) per una misura, in ordine di posizione."""
        scores = {
            pos: _tilt(w, syncopation) * (1.0 if _is_binary(pos) else tuplet_mix)
            for pos, w in self._weights.items()
        }
        total = sum(scores.values())
        if total == 0:
            return {pos: 0.0 for pos in scores}
        scale = density * self.beats / total
        return {pos: min(1.0, s * scale) for pos, s in scores.items()}

    def measure(
        self,
        rng: random.Random,
        *,
        density: float = 1.0,
        syncopation: float = 0.0,
        tuplet_mix: float = 0.0,
    ) -> list[tuple[Fraction, float]]:
        """Una misura: lista di (posizione relativa alla misura, velocity).

        Velocity = profilo inclinato w'(pos) in [0,1] (D7: accenti dallo
        stesso profilo delle probabilità).
        """
        probs = self.probabilities(
            density=density, syncopation=syncopation, tuplet_mix=tuplet_mix
        )
        return [
            (pos, _tilt(self._weights[pos], syncopation))
            for pos, p in probs.items()
            if rng.random() < p
        ]

    def sequence(
        self,
        n_measures: int,
        rng: random.Random,
        *,
        density: float = 1.0,
        syncopation: float = 0.0,
        tuplet_mix: float = 0.0,
    ) -> list[tuple[Fraction, float]]:
        """Sequenza open-loop a parametri fissi; posizioni assolute in beat."""
        events: list[tuple[Fraction, float]] = []
        for m in range(n_measures):
            offset = Fraction(m * self.beats)
            events.extend(
                (offset + pos, vel)
                for pos, vel in self.measure(
                    rng,
                    density=density,
                    syncopation=syncopation,
                    tuplet_mix=tuplet_mix,
                )
            )
        return events
