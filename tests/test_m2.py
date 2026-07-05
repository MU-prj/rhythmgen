"""Test di M2 (design D11): ascoltatore oracolo, valuta di sorpresa,
ancoraggio esterno a Longuet-Higgins & Lee e Toussaint (D10)."""

from fractions import Fraction as F
from statistics import correlation

import pytest

from rhythmgen import render
from rhythmgen.listener import (
    Level,
    OracleListener,
    default_levels,
    leaky_rms,
    pulse,
)
from rhythmgen.syncopation import lhl_syncopation, toussaint_complexity

# Batteria di pattern sulla griglia di 16 sedicesimi in 4/4 (D10).
PATTERNS = {
    "four_on_floor": [0, 4, 8, 12],
    "eighth_run": [0, 2, 4, 6, 8, 10, 12, 14],
    "backbeat": [4, 12],
    "tresillo": [0, 6, 12],
    "shiko": [0, 4, 6, 10, 12],
    "son": [0, 3, 6, 10, 12],
    "rumba": [0, 3, 7, 10, 12],
    "bossa": [0, 3, 6, 10, 13],
    "gahu": [0, 3, 6, 10, 14],
    "soukous": [0, 3, 6, 10, 11],
    "offbeat_eighths": [2, 6, 10, 14],
}


def oracle_score(onsets_16, **kwargs):
    events = render([(F(k, 4), 1.0) for k in sorted(onsets_16)], bpm=120.0)
    oracle = OracleListener(beats=4, bpm=120.0, **kwargs)
    return sum(e.surprise for e in oracle.listen(events, n_measures=1))


def test_pulse_forma_e_supporto():
    w = 0.25
    assert pulse(0.0, w) == 1.0
    assert pulse(w, w) == 0.0
    assert pulse(w * 2, w) == 0.0  # fuori dal supporto
    assert pulse(0.05, w) > pulse(0.10, w) > pulse(0.20, w)  # monotono
    assert pulse(-0.1, w) == pulse(0.1, w)  # simmetrico


def test_pesi_livelli_normalizzati():
    lvls = default_levels(4)
    assert [lv.period for lv in lvls] == [F(4), F(2), F(1), F(1, 2), F(1, 4)]
    assert sum(lv.weight for lv in lvls) == pytest.approx(1.0)
    # pesi custom non normalizzati vengono normalizzati dal listener
    oracle = OracleListener(4, 120.0, levels=[Level(F(1), 3.0), Level(F(4), 1.0)])
    assert sum(lv.weight for lv in oracle.levels) == pytest.approx(1.0)


def test_scala_di_sorpresa_per_onset():
    """La sorpresa di un onset riproduce la gerarchia LHL: battere < metà
    misura < beat < ottavo in levare < sedicesimo (D4)."""
    oracle = OracleListener(beats=4, bpm=120.0)

    def s(pos):
        [ev] = oracle.listen(render([(pos, 1.0)], bpm=120.0), n_measures=0)
        return ev.surprise

    assert s(F(0)) == pytest.approx(0.0)
    assert s(F(2)) == pytest.approx(0.2)
    assert s(F(1)) == pytest.approx(0.4)
    assert s(F(1, 2)) == pytest.approx(0.6)
    assert s(F(1, 4)) == pytest.approx(0.8)


def test_gerarchia_delle_omissioni():
    """Saltare il battere sorprende più che saltare un levare (vincolo D4):
    l'omissione somma i pesi di tutti i livelli con picco in quel punto."""
    oracle = OracleListener(beats=4, bpm=120.0)
    scored = oracle.listen([], n_measures=1)
    assert all(e.kind == "omission" for e in scored)
    by_pos = {e.position: e.surprise for e in scored}
    assert by_pos[F(0)] == pytest.approx(1.0)  # tutti e 5 i livelli
    assert by_pos[F(2)] == pytest.approx(0.8)
    assert by_pos[F(1)] == pytest.approx(0.6)
    assert by_pos[F(1, 2)] == pytest.approx(0.4)
    assert by_pos[F(1, 4)] == pytest.approx(0.2)
    assert F(1, 8) not in by_pos  # nessun picco → nessuna omissione
    # misura vuota: massa totale = Σ_livelli picchi·peso = (1+2+4+8+16)/5
    assert sum(e.surprise for e in scored) == pytest.approx(6.2)


def test_anticipazione_del_battere_conta_doppio():
    """Un onset un sedicesimo prima del battere non aggancia il picco:
    sorpresa dell'onset fuori posto + omissione del battere."""
    oracle = OracleListener(beats=4, bpm=120.0)
    events = render([(F(15, 4), 1.0)], bpm=120.0)
    scored = oracle.listen(events, n_measures=1)
    kinds = {(e.kind, e.position) for e in scored}
    assert ("onset", F(15, 4)) in kinds
    assert ("omission", F(0)) in kinds


def test_eventi_ordinati_e_sorprese_in_range():
    events = render([(F(k, 4), 1.0) for k in PATTERNS["son"]], bpm=120.0)
    scored = OracleListener(4, 120.0).listen(events, n_measures=1)
    assert all(0.0 <= e.surprise <= 1.0 for e in scored)
    times = [e.time_s for e in scored]
    assert times == sorted(times)
    assert {e.kind for e in scored} == {"onset", "omission"}


def test_lhl_valori_di_riferimento():
    assert lhl_syncopation(PATTERNS["four_on_floor"]) == 0
    assert lhl_syncopation(PATTERNS["eighth_run"]) == 0
    assert lhl_syncopation(PATTERNS["tresillo"]) == 2
    assert lhl_syncopation(PATTERNS["backbeat"]) == 3
    assert lhl_syncopation(PATTERNS["son"]) == 4
    assert lhl_syncopation(PATTERNS["rumba"]) == 6
    assert lhl_syncopation(PATTERNS["offbeat_eighths"]) == 7
    assert lhl_syncopation([]) == 0


def test_toussaint_valori_di_riferimento():
    assert toussaint_complexity(PATTERNS["four_on_floor"]) == 0
    assert toussaint_complexity(PATTERNS["eighth_run"]) == 0
    assert toussaint_complexity(PATTERNS["tresillo"]) == 2
    assert toussaint_complexity(PATTERNS["son"]) == 4
    assert toussaint_complexity(PATTERNS["offbeat_eighths"]) == 7


def test_ancoraggio_esterno_spearman():
    """D10: la sorpresa dell'oracolo deve correlare con gli indici simbolici
    (soglia di design 0.7; i valori misurati sono ≈0.98)."""
    names = list(PATTERNS)
    osc = [oracle_score(PATTERNS[n]) for n in names]
    lhl = [lhl_syncopation(PATTERNS[n]) for n in names]
    tou = [toussaint_complexity(PATTERNS[n]) for n in names]
    assert correlation(osc, lhl, method="ranked") > 0.9
    assert correlation(osc, tou, method="ranked") > 0.9


def test_ordinamento_musicologico():
    four = oracle_score(PATTERNS["four_on_floor"])
    tresillo = oracle_score(PATTERNS["tresillo"])
    son = oracle_score(PATTERNS["son"])
    offbeat = oracle_score(PATTERNS["offbeat_eighths"])
    assert four < tresillo < son < offbeat
    assert four == min(oracle_score(p) for p in PATTERNS.values())


def test_leaky_rms_converge_e_decade():
    from rhythmgen.listener import ScoredEvent

    costante = [
        ScoredEvent(t * 0.5, F(0), "onset", 0.4, {}) for t in range(40)
    ]
    trace = leaky_rms(costante, tau_s=2.0)
    assert trace[-1][1] == pytest.approx(0.4, abs=1e-6)
    # dopo un gradino a zero la traccia decade verso zero
    gradino = costante + [
        ScoredEvent(20.0 + t * 0.5, F(0), "onset", 0.0, {}) for t in range(40)
    ]
    trace = leaky_rms(gradino, tau_s=2.0)
    assert trace[-1][1] < 0.01
