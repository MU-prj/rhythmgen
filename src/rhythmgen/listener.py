"""Ascoltatore interno: valuta di sorpresa e modalità oracolo (design D2-D4).

Contratto comune alle due modalità (D2): un ascoltatore espone
``listen(events, n_measures) -> list[ScoredEvent]``. La sorpresa di ogni
evento è la somma pesata Σ wᵢ·Sᵢ sui livelli metrici, con Sᵢ ∈ [0,1] nella
valuta unica dell'aspettativa (D4): per un onset Sᵢ = 1 − Eᵢ(φ), per
un'omissione Sᵢ = Eᵢ al picco attraversato senza evento. L'oracolo legge la
fase metrica esatta da ``Event.position`` (M2); l'ascoltatore cieco (M3) la
inferirà dai soli ``Event.onset_s``.

La velocity non pesa ancora sulla sorpresa; la coerenza di fase è banale per
l'oracolo (fase esatta) e compare solo con l'ascoltatore cieco.
"""

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Optional

from .render import Event


def pulse(delta_beats: float, width_beats: float) -> float:
    """Pulse attenzionale a coseno rialzato nel dominio del tempo (D4).

    ``delta_beats``: distanza circolare dal picco, in beat. La larghezza è
    costante in tempo, non in frazione di ciclo: un livello lento ha un pulse
    proporzionalmente più stretto nella sua fase, altrimenti la vicinanza
    temporale al battere gonfia l'aspettativa dei levare (l'attenzione è
    concentrata sul punto atteso, non spalmata sul ciclo).
    """
    d = abs(delta_beats)
    if d >= width_beats:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * d / width_beats))


@dataclass(frozen=True)
class Level:
    period: Fraction  # in beat
    weight: float


def default_levels(beats: int) -> list[Level]:
    """Gerarchia binaria standard: misura, metà misura (se pari), beat,
    ottavi, sedicesimi, a pesi uguali.

    Con pesi uguali l'aspettativa totale sulle posizioni della griglia
    binaria riproduce la scala metrica di Longuet-Higgins & Lee (la
    prominenza di una posizione = numero di livelli che vi hanno il picco).
    """
    periods = [Fraction(beats)]
    if beats % 2 == 0:
        periods.append(Fraction(beats, 2))
    periods += [Fraction(1), Fraction(1, 2), Fraction(1, 4)]
    w = 1.0 / len(periods)
    return [Level(p, w) for p in periods]


@dataclass(frozen=True)
class ScoredEvent:
    time_s: float
    # onset: posizione dell'evento; omissione: posizione del picco.
    # None per l'ascoltatore cieco (M3), che le posizioni non le conosce.
    position: Optional[Fraction]
    kind: str  # "onset" | "omission"
    surprise: float  # Σ wᵢ·Sᵢ ∈ [0,1]
    per_level: dict[Fraction, float]  # periodo → Sᵢ non pesata


class OracleListener:
    """Modalità oracolo (D2): conosce griglia e tempo, nessuna inferenza.

    La fase di ogni livello è letta esattamente da ``Event.position``; le
    omissioni sono i picchi dei livelli attraversati senza un onset nella
    stessa identica posizione (aggancio esatto: le posizioni sono razionali,
    D6). Saltare il battere somma i pesi di tutti i livelli che vi hanno il
    picco, quindi sorprende più che saltare un levare (vincolo D4).
    """

    def __init__(
        self,
        beats: int,
        bpm: float,
        levels: Optional[Iterable[Level]] = None,
        pulse_width_beats: float = 0.25,
    ):
        self.beats = beats
        self.bpm = bpm
        lvls = list(levels) if levels is not None else default_levels(beats)
        total_w = sum(lv.weight for lv in lvls)
        # pesi normalizzati: la sorpresa di ogni evento resta in [0,1] (D4)
        self.levels = [Level(lv.period, lv.weight / total_w) for lv in lvls]
        self.width = pulse_width_beats

    def expectancies(self, pos: Fraction) -> dict[Fraction, float]:
        """Eᵢ(pos) per livello, valutando il pulse alla distanza circolare."""
        out = {}
        for lv in self.levels:
            phi = pos % lv.period
            delta = min(phi, lv.period - phi)
            out[lv.period] = pulse(float(delta), self.width)
        return out

    def listen(
        self,
        events: Iterable[Event],
        n_measures: int,
        start_measure: int = 0,
    ) -> list[ScoredEvent]:
        """Valuta gli eventi e le omissioni dei picchi nella finestra di
        ``n_measures`` misure a partire da ``start_measure``. La finestra
        serve al loop chiuso (M4), che ascolta una misura alla volta:
        l'oracolo è senza stato, quindi finestre consecutive equivalgono
        all'ascolto dell'intera sequenza."""
        beat_s = 60.0 / self.bpm
        scored = []
        onset_positions = set()
        for ev in events:
            per_level = {p: 1.0 - e for p, e in self.expectancies(ev.position).items()}
            surprise = sum(lv.weight * per_level[lv.period] for lv in self.levels)
            scored.append(ScoredEvent(ev.onset_s, ev.position, "onset", surprise, per_level))
            onset_positions.add(ev.position)

        lo = Fraction(start_measure * self.beats)
        hi = lo + n_measures * self.beats
        peaks: dict[Fraction, list[Level]] = {}
        for lv in self.levels:
            k = math.ceil(lo / lv.period)
            while k * lv.period < hi:
                peaks.setdefault(k * lv.period, []).append(lv)
                k += 1
        for pos, lvls in peaks.items():
            if pos in onset_positions:
                continue
            per_level = {lv.period: 1.0 for lv in lvls}  # Sᵢ = Eᵢ(picco) = 1
            surprise = sum(lv.weight for lv in lvls)
            scored.append(
                ScoredEvent(float(pos) * beat_s, pos, "omission", surprise, per_level)
            )
        return sorted(scored, key=lambda e: e.time_s)


class SurpriseRMS:
    """Integratore incrementale della valuta (D4): media mobile RMS a
    decadimento esponenziale, y ← y·α + s²·(1−α) con α = exp(−Δt/τ).
    Con sorpresa costante s il valore converge a s.

    Forma a stato di ``leaky_rms``, per il loop chiuso (M4): gli
    aggiornamenti arrivano in ordine di tempo; un Δt negativo (omissione
    emessa in ritardo dall'ascoltatore cieco) è trattato come simultaneo.
    """

    def __init__(self, tau_s: float):
        self.tau_s = tau_s
        self._y = 0.0
        self._t_prev: Optional[float] = None

    @property
    def value(self) -> float:
        return math.sqrt(self._y)

    def update(self, t_s: float, surprise: float) -> float:
        if self._t_prev is None:
            alpha = 0.0
            self._t_prev = t_s
        else:
            alpha = math.exp(-max(t_s - self._t_prev, 0.0) / self.tau_s)
            self._t_prev = max(t_s, self._t_prev)
        self._y = self._y * alpha + surprise**2 * (1.0 - alpha)
        return math.sqrt(self._y)


def leaky_rms(scored: Iterable[ScoredEvent], tau_s: float) -> list[tuple[float, float]]:
    """Media mobile RMS a decadimento esponenziale, campionata agli eventi (D4)."""
    rms = SurpriseRMS(tau_s)
    return [
        (ev.time_s, rms.update(ev.time_s, ev.surprise))
        for ev in sorted(scored, key=lambda e: e.time_s)
    ]
