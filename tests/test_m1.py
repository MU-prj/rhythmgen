"""Test open-loop di M1 (design D11): griglia, generatore, rendering."""

import json
import random
from fractions import Fraction as F

from rhythmgen import (
    BernoulliGenerator,
    build_grid,
    events_to_json,
    metric_weight,
    render,
)


def test_grid_unione_sedicesimi_terzine():
    grid = build_grid(2, {4, 3})
    dentro_beat0 = {F(0), F(1, 4), F(1, 3), F(1, 2), F(2, 3), F(3, 4)}
    assert {p for p in grid if p < 1} == dentro_beat0
    assert len(grid) == 12
    assert grid == sorted(grid)
    assert all(isinstance(p, F) and 0 <= p < 2 for p in grid)


def test_pesi_gerarchia_notazionale():
    w = lambda p: metric_weight(p, 4)
    assert w(F(0)) == 1.0
    assert w(F(2)) == 0.8  # metà misura
    assert w(F(1)) == w(F(3)) == 0.6
    # battere > metà misura > beat > ottavo > terzina > sedicesimo > quintina
    assert w(F(0)) > w(F(2)) > w(F(1)) > w(F(1, 2)) > w(F(1, 3)) > w(F(1, 4)) > w(F(1, 5))


def test_somma_probabilita_uguale_density_per_beats():
    gen = BernoulliGenerator(4, {4})
    probs = gen.probabilities(density=0.5, syncopation=0.2)
    assert abs(sum(probs.values()) - 0.5 * 4) < 1e-9  # Σp = density·beats (D7)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_frequenze_empiriche_seguono_profilo_e_inversione():
    gen = BernoulliGenerator(4, {2})
    n = 4000

    def freq(syncopation):
        rng = random.Random(42)
        counts = {p: 0 for p in gen.grid}
        for _ in range(n):
            for pos, _ in gen.measure(rng, density=0.5, syncopation=syncopation):
                counts[pos] += 1
        return {p: c / n for p, c in counts.items()}

    dritto = freq(0.0)
    assert dritto[F(0)] > dritto[F(1)] > dritto[F(1, 2)]  # battere > beat > levare
    inverso = freq(1.0)
    assert inverso[F(1, 2)] > inverso[F(1)] > inverso[F(0)]  # profilo invertito


def test_density_media_eventi_per_misura():
    gen = BernoulliGenerator(4, {4, 3})
    rng = random.Random(7)
    n = 2000
    tot = sum(
        len(gen.measure(rng, density=0.8, syncopation=0.3, tuplet_mix=0.5))
        for _ in range(n)
    )
    assert abs(tot / n - 0.8 * 4) < 0.1


def test_tuplet_mix_zero_esclude_irregolari():
    gen = BernoulliGenerator(4, {4, 3})
    rng = random.Random(5)
    seq = gen.sequence(200, rng, density=1.0, syncopation=0.5, tuplet_mix=0.0)
    assert seq, "a density 1.0 la sequenza non può essere vuota"
    assert all((pos % 1).denominator != 3 for pos, _ in seq)


def test_posizioni_dal_grid_niente_float_spurii():
    gen = BernoulliGenerator(4, {4, 3})
    rng = random.Random(1)
    seq = gen.sequence(50, rng, density=1.0, syncopation=0.5, tuplet_mix=1.0)
    grid_set = set(gen.grid)
    for pos, vel in seq:
        assert isinstance(pos, F)
        assert pos % gen.beats in grid_set
        assert 0.0 <= vel <= 1.0


def test_render_deterministico_e_jitter_limitato():
    events = [(F(0), 1.0), (F(1, 2), 0.4), (F(1), 0.6)]
    esatto = render(events, bpm=120.0)  # beat = 0.5 s
    assert [e.onset_s for e in esatto] == [0.0, 0.25, 0.5]
    assert [e.position for e in esatto] == [F(0), F(1, 2), F(1)]

    jittered = render(events, bpm=120.0, sigma_t=0.01, rng=random.Random(3))
    assert any(j.onset_s != e.onset_s for j, e in zip(jittered, esatto))
    assert all(abs(j.onset_s - e.onset_s) < 0.05 for j, e in zip(jittered, esatto))


def test_determinismo_stesso_seed():
    gen = BernoulliGenerator(4, {4})
    a = gen.sequence(20, random.Random(9), density=0.7, syncopation=0.4)
    b = gen.sequence(20, random.Random(9), density=0.7, syncopation=0.4)
    assert a == b


def test_json_round_trip():
    out = render([(F(7, 3), 0.5)], bpm=60.0)
    data = json.loads(events_to_json(out))
    assert data == [{"onset_s": 7 / 3, "position": "7/3", "velocity": 0.5}]
