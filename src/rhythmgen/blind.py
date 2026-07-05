"""Ascoltatore cieco: banco di oscillatori adattivi (design D2, D3; M3).

Riceve SOLI timestamp (``Event.onset_s``): periodo e fase sono inferiti e
possono sbagliare — perdere il beat, o ri-agganciarsi a un metro spostato.
La divergenza dall'oracolo è il segnale di difficoltà inferenziale della
sequenza (D2), non un errore del modulo.

Accoppiamento "a rapporti vincolati" (scelta M3 tra le opzioni lasciate
aperte dal design):

- un periodo master τ (il tactus) e un'ancora di fase A, adattati dagli
  onset con correzioni pesate da un kernel di accoppiamento (PLL del
  secondo ordine);
- i livelli hanno periodi a rapporto FISSO col tactus: la forma del metro è
  un prior strutturale, tempo e fase restano inferiti; i livelli sopra il
  beat (misura, metà misura) hanno un offset di fase proprio, adattato più
  lentamente (D3), che permette di re-inferire dove cade il battere;
- i livelli sotto il beat ereditano l'ancora del tactus (una suddivisione
  che deriva dal proprio beat non è un percetto: è incoerenza).

Il kernel di accoppiamento è più largo del pulse di aspettativa: un evento
lontano dal picco sorprende (E ≈ 0) ma trascina comunque un poco
l'oscillatore. È ciò che rende possibile il ri-aggancio a un profilo
spostato — l'assuefazione emergente dell'ipotesi centrale del design.

Il warmup (i primi ``bootstrap_onsets`` onset) inizializza τ con la mediana
degli inter-onset interval e assume il primo onset come battere; gli eventi
di warmup non producono ScoredEvent.

Due interfacce sullo stesso stato (M4): ``listen(events, duration_s)`` batch,
e la coppia incrementale ``feed(t)`` / ``finish(duration_s)`` per il loop
chiuso, che riceve gli onset man mano che il generatore li produce. Il batch
è un wrapper della coppia: stesso percorso di codice, stessi risultati.
"""

import math
from statistics import median
from typing import Iterable, Optional

from .listener import Level, ScoredEvent, default_levels, pulse
from .render import Event


class BlindListener:
    """Modalità cieca del contratto D2. ``listen(events, duration_s)``
    restituisce ScoredEvent con ``position=None`` (le posizioni non le
    conosce); ``duration_s`` serve solo a emettere le omissioni finali.

    Dopo ``listen()``: ``tau_s`` è il periodo inferito del tactus e
    ``coherence_trace`` la coerenza di fase del tactus nel tempo (EMA
    complessa del vettore di fase, in [0,1])."""

    def __init__(
        self,
        beats: int = 4,
        levels: Optional[Iterable[Level]] = None,
        pulse_width_beats: float = 0.25,
        coupling_width_factor: float = 0.6,
        eta_phase: float = 0.15,
        eta_period: float = 0.02,
        bootstrap_onsets: int = 4,
        coherence_tau_s: float = 4.0,
    ):
        lvls = list(levels) if levels is not None else default_levels(beats)
        total_w = sum(lv.weight for lv in lvls)
        self.levels = [Level(lv.period, lv.weight / total_w) for lv in lvls]
        self.width = pulse_width_beats
        self.coupling_factor = coupling_width_factor
        self.eta_phase = eta_phase
        self.eta_period = eta_period
        self.bootstrap_onsets = bootstrap_onsets
        self.coherence_tau_s = coherence_tau_s
        self._reset()

    def _reset(self) -> None:
        self.tau_s: Optional[float] = None
        self.coherence_trace: list[tuple[float, float]] = []
        self._boot: list[float] = []
        self._dead = False  # warmup degenere (IOI tutti nulli): niente tempo
        self._z = 0j
        self._t_prev: Optional[float] = None

    # --- geometria dei picchi ------------------------------------------
    def _peak_time(self, i: int, k: int) -> float:
        return self._anchor + self._offsets[i] + k * float(self.levels[i].period) * self.tau_s

    def _nearest(self, i: int, t: float) -> tuple[int, float]:
        """(k, δ_s) del picco più vicino a t per il livello i. I tie in
        controfase esatta si rompono sempre verso l'alto (floor(x+0.5)):
        il trascinamento resta di segno coerente e il ri-aggancio può
        partire anche da un flusso perfettamente in levare."""
        P = float(self.levels[i].period) * self.tau_s
        k = math.floor((t - self._anchor - self._offsets[i]) / P + 0.5)
        return k, t - self._peak_time(i, k)

    def _flush_omissions(self, t: float, scored: list[ScoredEvent]) -> None:
        """Emette le omissioni per ogni picco passato senza onset agganciato
        (Sᵢ = Eᵢ al picco = 1, pesata; D4)."""
        dock = self.width * self.tau_s
        for i, lv in enumerate(self.levels):
            k = self._flushed[i] + 1
            while self._peak_time(i, k) + dock < t:
                if k not in self._docked[i]:
                    scored.append(
                        ScoredEvent(self._peak_time(i, k), None, "omission",
                                    lv.weight, {lv.period: 1.0})
                    )
                self._docked[i].discard(k)
                self._flushed[i] = k
                k += 1

    # --- ascolto --------------------------------------------------------
    def feed(self, t: float) -> list[ScoredEvent]:
        """Un onset alla volta, in ordine di tempo (interfaccia per il loop
        chiuso, M4). Restituisce gli ScoredEvent emessi da questo onset: le
        omissioni dei picchi ormai passati più l'onset stesso; lista vuota
        durante il warmup."""
        if self._dead:
            return []
        if self.tau_s is None:
            if len(self._boot) < self.bootstrap_onsets:
                self._boot.append(t)
                return []  # il warmup non produce eventi valutati
            boot = self._boot
            iois = [b - a for a, b in zip(boot, boot[1:]) if b > a]
            if not iois:
                self._dead = True
                return []
            self.tau_s = median(iois)
            self._anchor = boot[0]  # primo onset = beat, e battere assunto
            self._offsets = [0.0] * len(self.levels)
            self._docked: list[set[int]] = [set() for _ in self.levels]
            # i picchi fino alla fine del warmup sono già consumati
            self._flushed = [
                math.floor((boot[-1] - self._anchor) / (float(lv.period) * self.tau_s))
                for lv in self.levels
            ]

        scored: list[ScoredEvent] = []
        self._flush_omissions(t, scored)

        # valuta PRIMA di adattare: la sorpresa riflette l'aspettativa
        # formata sul passato (D4)
        snapshot = []
        per_level: dict = {}
        for i, lv in enumerate(self.levels):
            k, d = self._nearest(i, t)
            E = pulse(d / self.tau_s, self.width)
            snapshot.append((i, lv, k, d, E))
            per_level[lv.period] = 1.0 - E
        surprise = sum(lv.weight * per_level[lv.period] for lv in self.levels)
        scored.append(ScoredEvent(t, None, "onset", surprise, per_level))

        theta = 0.0
        for i, lv, k, d, E in snapshot:
            if E > 0.0 and k > self._flushed[i]:
                self._docked[i].add(k)
            # adattamento PLL, pesato dal kernel di accoppiamento; il
            # kernel ha larghezza costante in tempo come il pulse (D4):
            # se scalasse col periodo, il livello misura adatterebbe più
            # in fretta del tactus, l'opposto del design (D3). Con il
            # kernel locale i livelli lenti scivolano insieme all'ancora
            # del tactus e re-inferiscono il battere solo su evidenza
            # vicina ai loro picchi.
            kappa = pulse(d / self.tau_s, self.coupling_factor)
            if lv.period == 1:
                theta = d / self.tau_s
                self._anchor += self.eta_phase * kappa * d
                # ponytail: floor a 50 ms, un τ collassato non è un tempo
                self.tau_s = max(self.tau_s + self.eta_period * kappa * d, 0.05)
            elif lv.period > 1:
                self._offsets[i] += (self.eta_phase / float(lv.period)) * kappa * d

        lam = (
            math.exp(-(t - self._t_prev) / self.coherence_tau_s)
            if self._t_prev is not None
            else 0.0
        )
        self._z = self._z * lam + (1.0 - lam) * complex(
            math.cos(2 * math.pi * theta), math.sin(2 * math.pi * theta)
        )
        self._t_prev = t
        self.coherence_trace.append((t, abs(self._z)))
        return scored

    def finish(self, duration_s: Optional[float] = None) -> list[ScoredEvent]:
        """Chiude lo stream: le omissioni residue fino a ``duration_s``
        (default: l'ultimo onset ricevuto)."""
        if self.tau_s is None:
            return []
        scored: list[ScoredEvent] = []
        self._flush_omissions(
            duration_s if duration_s is not None else self._t_prev, scored
        )
        return scored

    def listen(self, events: Iterable[Event], duration_s: Optional[float] = None) -> list[ScoredEvent]:
        """Interfaccia batch del contratto D2: wrapper di feed/finish."""
        self._reset()
        scored: list[ScoredEvent] = []
        for t in sorted(e.onset_s for e in events):
            scored += self.feed(t)
        scored += self.finish(duration_s)
        return sorted(scored, key=lambda e: e.time_s)
