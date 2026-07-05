"""Controllore omeostatico e loop chiuso (design D5, D11; M4).

Il loop chiude generatore e ascoltatore: a ogni misura il controllore PI
legge la valuta integrata (SurpriseRMS, D4) e muove la sincopazione σ del
generatore — la leva primaria (D7) — per inseguire una traiettoria di
setpoint S_target(t). La traiettoria è una spezzata lineare (D5): un solo
meccanismo copre setpoint costante, rampa e arco narrativo. Il setpoint è
filtrato passa-basso e il controllore agisce una volta per misura: risponde
su scale musicali (secondi), più lente dell'adattamento dell'ascoltatore.

Ascoltatore nel loop: oracolo (baseline senza assuefazione) o cieco (il
regime dell'ipotesi centrale — l'assuefazione emerge dall'entrainment e il
controllore deve muoversi per sostenere la sorpresa).
"""

import json
import math
import random
from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Optional, Union

from .blind import BlindListener
from .generator import BernoulliGenerator
from .listener import OracleListener, ScoredEvent, SurpriseRMS
from .render import Event, render


class SetpointCurve:
    """Traiettoria S_target(t): spezzata lineare per punti (D5).

    Prima del primo punto e dopo l'ultimo la curva è costante (clamp).
    ``constant`` e ``ramp`` sono i casi d'uso nominati dal design.
    """

    def __init__(self, points: Iterable[tuple[float, float]]):
        pts = sorted(points)
        if not pts:
            raise ValueError("serve almeno un punto (t_s, valore)")
        if any(b[0] <= a[0] for a, b in zip(pts, pts[1:])):
            raise ValueError("i tempi dei punti devono essere strettamente crescenti")
        self._times = [t for t, _ in pts]
        self._values = [v for _, v in pts]

    @classmethod
    def constant(cls, value: float) -> "SetpointCurve":
        return cls([(0.0, value)])

    @classmethod
    def ramp(cls, v0: float, v1: float, t0: float, t1: float) -> "SetpointCurve":
        return cls([(t0, v0), (t1, v1)])

    def __call__(self, t_s: float) -> float:
        i = bisect_right(self._times, t_s)
        if i == 0:
            return self._values[0]
        if i == len(self._times):
            return self._values[-1]
        t0, t1 = self._times[i - 1], self._times[i]
        v0, v1 = self._values[i - 1], self._values[i]
        return v0 + (v1 - v0) * (t_s - t0) / (t1 - t0)


class LowPass:
    """Filtro passa-basso del primo ordine sul setpoint (D5).

    Si inizializza al primo campione: il transitorio iniziale non deve
    essere un artefatto del filtro che parte da zero.
    """

    def __init__(self, tau_s: float):
        self.tau_s = tau_s
        self._y: Optional[float] = None

    def update(self, x: float, dt_s: float) -> float:
        if self._y is None or self.tau_s <= 0:
            self._y = x
        else:
            self._y += (1.0 - math.exp(-dt_s / self.tau_s)) * (x - self._y)
        return self._y


class PIController:
    """PI con anti-windup (D5). L'uscita è direttamente il parametro
    controllato, limitata a [lo, hi]. Anti-windup per confinamento:
    l'integrale — il valore "di riposo" del parametro — vive anch'esso in
    [lo, hi], quindi la carica accumulata contro un setpoint irraggiungibile
    è limitata e all'inversione dell'errore il termine proporzionale stacca
    l'uscita dal bordo immediatamente.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        lo: float = 0.0,
        hi: float = 1.0,
        u0: float = 0.0,
    ):
        if not lo <= u0 <= hi:
            raise ValueError("u0 deve stare in [lo, hi]")
        self.kp = kp
        self.ki = ki
        self.lo = lo
        self.hi = hi
        self.integral = u0  # lo stato parte dal valore iniziale del parametro

    def step(self, error: float, dt_s: float) -> float:
        self.integral += self.ki * error * dt_s
        self.integral = min(max(self.integral, self.lo), self.hi)
        u = self.kp * error + self.integral
        return min(max(u, self.lo), self.hi)


@dataclass(frozen=True)
class TracePoint:
    """Un campione della firma dinamica, uno per misura (D9, D10)."""

    t_s: float  # inizio della misura
    target: float  # setpoint filtrato visto dal controllore
    sigma: float  # sincopazione applicata alla misura
    surprise_rms: float  # valuta integrata a fine misura


@dataclass(frozen=True)
class LoopResult:
    events: list[Event]
    scored: list[ScoredEvent]
    trace: list[TracePoint]


def trace_to_json(trace: Iterable[TracePoint]) -> str:
    """Serializzazione JSON della traccia (D9)."""
    return json.dumps(
        [
            {
                "t": p.t_s,
                "target": p.target,
                "sigma": p.sigma,
                "surprise_rms": p.surprise_rms,
            }
            for p in trace
        ],
        indent=2,
    )


class ClosedLoop:
    """Feedback loop misura-per-misura (M4).

    A ogni misura: campiona e filtra il setpoint, muove σ col PI
    sull'errore rispetto alla valuta integrata, genera la misura con la σ
    corrente, la rende in secondi e la fa ascoltare. Con ``controller=None``
    il loop gira ad anello aperto a ``sigma0`` costante: è il ramo
    "parametri congelati" del test dell'ipotesi centrale (D10).

    ``engage_measure``: il controllore si innesta solo da quella misura; le
    misure precedenti girano a ``sigma0``. Con l'ascoltatore cieco il
    prologo serve da warm-lock: prima che il PI muova σ ha senso che
    l'ascoltatore abbia agganciato qualcosa da cui essere sorpreso
    (l'ascoltatore si adatta più in fretta di quanto il controllore muova
    il generatore, D5).

    L'ascoltatore è un'istanza di OracleListener (finestre per misura) o di
    BlindListener (feed/finish incrementali). Il cieco emette le omissioni
    di una misura solo all'onset successivo: la valuta del loop è causale,
    come dev'essere per un controllore.
    """

    def __init__(
        self,
        generator: BernoulliGenerator,
        bpm: float,
        listener: Union[OracleListener, BlindListener],
        controller: Optional[PIController] = None,
        *,
        rms_tau_s: float = 3.0,
        lpf_tau_s: float = 2.0,
        density: float = 1.0,
        tuplet_mix: float = 0.0,
        sigma0: float = 0.0,
        sigma_t: float = 0.0,
        engage_measure: int = 0,
    ):
        self.generator = generator
        self.bpm = bpm
        self.listener = listener
        self.controller = controller
        self.rms = SurpriseRMS(rms_tau_s)
        self.lpf = LowPass(lpf_tau_s)
        self.density = density
        self.tuplet_mix = tuplet_mix
        self.sigma0 = sigma0
        self.sigma_t = sigma_t
        self.engage_measure = engage_measure

    def _score_measure(self, events: list[Event], m: int) -> list[ScoredEvent]:
        if isinstance(self.listener, BlindListener):
            scored: list[ScoredEvent] = []
            for ev in events:
                scored.extend(self.listener.feed(ev.onset_s))
            return scored
        return self.listener.listen(events, n_measures=1, start_measure=m)

    def run(
        self,
        n_measures: int,
        setpoint: SetpointCurve,
        rng: random.Random,
    ) -> LoopResult:
        beats = self.generator.beats
        meas_s = beats * 60.0 / self.bpm
        events_all: list[Event] = []
        scored_all: list[ScoredEvent] = []
        trace: list[TracePoint] = []
        sigma = self.sigma0

        for m in range(n_measures):
            t0 = m * meas_s
            target = self.lpf.update(setpoint(t0), meas_s)
            if self.controller is not None and m >= self.engage_measure:
                sigma = self.controller.step(target - self.rms.value, meas_s)

            offset = Fraction(m * beats)
            measure = self.generator.measure(
                rng,
                density=self.density,
                syncopation=sigma,
                tuplet_mix=self.tuplet_mix,
            )
            events = render(
                [(offset + pos, vel) for pos, vel in measure],
                self.bpm,
                sigma_t=self.sigma_t,
                rng=rng,
            )
            events.sort(key=lambda e: e.onset_s)

            scored = self._score_measure(events, m)
            for ev in sorted(scored, key=lambda e: e.time_s):
                self.rms.update(ev.time_s, ev.surprise)

            events_all.extend(events)
            scored_all.extend(scored)
            trace.append(TracePoint(t0, target, sigma, self.rms.value))

        if isinstance(self.listener, BlindListener):
            tail = self.listener.finish(n_measures * meas_s)
            for ev in sorted(tail, key=lambda e: e.time_s):
                self.rms.update(ev.time_s, ev.surprise)
            scored_all.extend(tail)

        scored_all.sort(key=lambda e: e.time_s)
        return LoopResult(events_all, scored_all, trace)
