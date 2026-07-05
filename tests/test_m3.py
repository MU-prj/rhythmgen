"""Test di M3 (design D11): ascoltatore cieco, convergenza verso l'oracolo
nel regime facile, ri-aggancio come dato (non bug), degrado con jitter."""

import random
from fractions import Fraction as F

import pytest

from rhythmgen import BernoulliGenerator, BlindListener, OracleListener, render

BPM = 120.0
BEAT_S = 60.0 / BPM  # 0.5 s
MEAS_S = 4 * BEAT_S  # 2.0 s


def per_measure(scored, n_measures):
    sums = [0.0] * n_measures
    for e in scored:
        sums[min(int(e.time_s / MEAS_S), n_measures - 1)] += e.surprise
    return sums


def listen_pair(positions, n_measures, sigma_t=0.0, rng=None):
    events = render(positions, bpm=BPM, sigma_t=sigma_t, rng=rng)
    oracle = OracleListener(4, BPM).listen(events, n_measures=n_measures)
    blind = BlindListener(4)
    scored = blind.listen(events, duration_s=n_measures * MEAS_S)
    return per_measure(oracle, n_measures), per_measure(scored, n_measures), blind


def test_metronomo_convergenza_stretta():
    """Regime facilissimo: il cieco replica l'oracolo (D11-M3)."""
    n = 30
    po, pb, blind = listen_pair([(F(k), 1.0) for k in range(n * 4)], n)
    assert blind.tau_s == pytest.approx(BEAT_S, rel=1e-6)
    assert blind.coherence_trace[-1][1] > 0.99
    late = sum(abs(a - b) for a, b in zip(po[24:], pb[24:])) / 6
    assert late < 0.1


def test_regime_facile_convergenza_statistica():
    """Generatore su soli beat (palette {1}), σ bassa: il cieco converge
    all'oracolo dopo il warmup (D11-M3, regime facile)."""
    n = 40
    gen = BernoulliGenerator(4, {1})
    seq = gen.sequence(n, random.Random(0), density=0.85, syncopation=0.1)
    po, pb, blind = listen_pair(seq, n)
    assert blind.tau_s == pytest.approx(BEAT_S, abs=0.005)
    assert blind.coherence_trace[-1][1] > 0.99
    late = sum(abs(a - b) for a, b in zip(po[30:], pb[30:])) / 10
    assert late < 0.1


def test_riaggancio_al_metro_spostato_e_il_dato():
    """Ipotesi centrale al livello dell'ascoltatore: dopo lo switch in
    levare l'oracolo resta sorpreso per sempre, il cieco spicca alla
    violazione e poi si ri-aggancia al metro spostato: l'assuefazione
    emerge dall'entrainment (D2, D11-M3)."""
    n = 32
    pos = [(F(k), 1.0) for k in range(8 * 4)]
    pos += [(F(k) + F(1, 2), 1.0) for k in range(8 * 4, n * 4)]
    po, pb, blind = listen_pair(pos, n)

    # prima dello switch entrambi tranquilli
    assert sum(pb[4:8]) / 4 == pytest.approx(sum(po[4:8]) / 4, abs=0.1)
    # l'oracolo resta alto per sempre (il levare non diventa mai il beat)
    assert sum(po[27:]) / 5 > 6.8
    # il cieco spicca alla violazione...
    assert pb[8] > 7.0
    # ...e a regime si è ri-agganciato: sorpresa da metronomo, non da levare
    late_blind = sum(pb[27:]) / 5
    assert late_blind < 5.0
    assert late_blind < sum(po[27:]) / 5 - 2.0
    # e il tactus inferito è tornato un periodo sensato
    assert blind.tau_s == pytest.approx(BEAT_S, rel=0.02)
    assert blind.coherence_trace[-1][1] > 0.99


def test_jitter_degrada_coerenza_e_alza_sorpresa():
    """D11-M3: con jitter crescente il cieco degrada rispetto a se stesso
    (coerenza giù, sorpresa a regime su); il τ resta agganciato."""
    n = 30
    pos = [(F(k), 1.0) for k in range(n * 4)]

    def run(sig):
        _, pb, blind = listen_pair(pos, n, sigma_t=sig, rng=random.Random(5))
        return blind.coherence_trace[-1][1], sum(pb[24:]) / 6, blind.tau_s

    c0, s0, t0 = run(0.0)
    c2, s2, t2 = run(0.02)
    c4, s4, t4 = run(0.04)
    assert c0 > c2 + 0.2 and c2 > c4 + 0.05
    assert s0 < s2 < s4
    for t in (t0, t2, t4):
        assert t == pytest.approx(BEAT_S, rel=0.02)


def test_beat_mancante_genera_omissione():
    """Un buco nel metronomo produce omissioni alla posizione attesa, con
    massa ≈ 1 (tutti i livelli hanno il picco sul battere saltato)."""
    n = 20
    missing = F(10 * 4)  # il battere della misura 10
    pos = [(F(k), 1.0) for k in range(n * 4) if F(k) != missing]
    _, _, blind = listen_pair(pos, n)  # per lo stato
    events = render(pos, bpm=BPM)
    scored = BlindListener(4).listen(events, duration_s=n * MEAS_S)
    t_miss = float(missing) * BEAT_S
    hole = [
        e for e in scored
        if e.kind == "omission" and abs(e.time_s - t_miss) < 0.13 * BEAT_S
    ]
    assert sum(e.surprise for e in hole) == pytest.approx(1.0, abs=0.05)
    assert blind.coherence_trace[-1][1] > 0.99  # il buco non destabilizza


def test_ambiguita_di_livello_metrico():
    """Flusso isocrono di sedicesimi: il cieco aggancia COERENTEMENTE il
    livello sbagliato (tactus = sedicesimo). La divergenza dall'oracolo
    misura la scelta di livello, non un fallimento (nota di design M3)."""
    n = 12
    pos = [(F(k, 4), 1.0) for k in range(n * 16)]
    po, pb, blind = listen_pair(pos, n)
    assert blind.tau_s == pytest.approx(BEAT_S / 4, rel=0.02)
    assert blind.coherence_trace[-1][1] > 0.9
    assert abs(sum(pb[8:]) - sum(po[8:])) / 4 > 3.0  # divergenza strutturale


def test_contratto_cieco():
    """Contratto D2: position=None, kinds noti, sorprese in [0,1], tempi
    ordinati, warmup senza eventi valutati, stream corti → lista vuota."""
    n = 10
    gen = BernoulliGenerator(4, {2})
    seq = gen.sequence(n, random.Random(3), density=1.0, syncopation=0.3)
    events = render(seq, bpm=BPM)
    blind = BlindListener(4)
    scored = blind.listen(events, duration_s=n * MEAS_S)
    assert all(e.position is None for e in scored)
    assert {e.kind for e in scored} <= {"onset", "omission"}
    assert all(0.0 <= e.surprise <= 1.0 for e in scored)
    times = [e.time_s for e in scored]
    assert times == sorted(times)
    n_onsets = sum(1 for e in scored if e.kind == "onset")
    assert n_onsets == len(events) - blind.bootstrap_onsets
    periods = {lv.period for lv in blind.levels}
    assert all(set(e.per_level) <= periods for e in scored)
    # stream più corto del warmup: niente da inferire
    assert BlindListener(4).listen(events[:4], duration_s=8.0) == []
