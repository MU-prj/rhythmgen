"""Griglia metrica razionale e pesi metrici a priori (design D6, D7).

Convenzione di unità (D6): le posizioni sono frazioni di beat, con beat = 1;
la misura è un numero intero di beat.
"""

from fractions import Fraction
from typing import Iterable


def build_grid(beats: int, subdivisions: Iterable[int]) -> list[Fraction]:
    """Posizioni ammesse in una misura di ``beats`` beat.

    ``subdivisions`` è la palette piatta: parti per beat di ogni suddivisione
    attiva (es. ``{4, 3}`` = sedicesimi + terzine di ottavi). L'unione delle
    suddivisioni genera l'insieme delle posizioni.
    """
    subs = set(subdivisions)
    if beats < 1 or not subs or any(d < 1 for d in subs):
        raise ValueError("beats e suddivisioni devono essere interi >= 1")
    return sorted(
        {
            Fraction(b) + Fraction(k, d)
            for b in range(beats)
            for d in subs
            for k in range(d)
        }
    )


def metric_weight(pos: Fraction, beats: int) -> float:
    """Peso a priori dalla gerarchia notazionale (D7).

    Battere di misura > metà misura > beat > ottavi > terzine > sedicesimi >
    suddivisioni più fini. Posizioni coincidenti tra suddivisioni (es. 2/4 e
    1/2) ricevono il peso del livello più forte perché la Fraction è ridotta.
    """
    if pos == 0:
        return 1.0
    if beats % 2 == 0 and pos == Fraction(beats, 2):
        return 0.8
    if pos.denominator == 1:
        return 0.6
    within_beat = pos % 1
    return {2: 0.4, 3: 0.3, 4: 0.25}.get(within_beat.denominator, 0.2)
