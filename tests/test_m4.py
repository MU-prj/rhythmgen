"""Test di M4 (design D11): controllore PI, traiettoria di setpoint,
omeostasi e inseguimento closed-loop, test dell'ipotesi centrale (D10).

Le soglie dei test stocastici sono calibrate su seed fissi (prassi D10):
i seed dei test col cieco sono scelti tra quelli il cui warmup aggancia
(regime facile del bootstrap, nota M3); il comportamento closed-loop
successivo è ciò che il test misura, non ciò che seleziona.
"""

import json
import random
import statistics
from fractions import Fraction as F

import pytest

from rhythmgen import (
    BernoulliGenerator,
    BlindListener,
    ClosedLoop,
    LowPass,
    OracleListener,
    PIController,
    SetpointCurve,
    SurpriseRMS,
    leaky_rms,
    render,
    trace_to_json,
)
from rhythmgen.listener import ScoredEvent

BPM = 120.0
MEAS_S = 2.0  # 4 beat a 120 BPM


def make_loop(kind, *, kp=2.0, ki=0.4, u0=0.1, sigma0=0.1, engage=12):
    gen = BernoulliGenerator(4, {2})
    listener = OracleListener(4, BPM) if kind == "oracle" else BlindListener(4)
    ctl = PIController(kp, ki, u0=u0)
    return ClosedLoop(
        gen, BPM, listener, ctl, sigma0=sigma0, engage_measure=engage
    )


# --- traiettoria di setpoint (D5) --------------------------------------


def test_setpoint_costante_rampa_spezzata():
    const = SetpointCurve.constant(0.4)
    assert const(-5.0) == const(0.0) == const(100.0) == 0.4

    ramp = SetpointCurve.ramp(0.2, 0.6, 10.0, 30.0)
    assert ramp(0.0) == 0.2  # clamp prima dell'inizio
    assert ramp(10.0) == 0.2
    assert ramp(20.0) == pytest.approx(0.4)
    assert ramp(30.0) == ramp(99.0) == 0.6  # clamp dopo la fine

    arco = SetpointCurve([(0.0, 0.3), (10.0, 0.5), (20.0, 0.2)])
    assert arco(5.0) == pytest.approx(0.4)
    assert arco(15.0) == pytest.approx(0.35)

    with pytest.raises(ValueError):
        SetpointCurve([])
    with pytest.raises(ValueError):
        SetpointCurve([(0.0, 0.1), (0.0, 0.2)])  # tempi non crescenti


def test_lowpass_si_inizializza_al_primo_campione():
    lpf = LowPass(2.0)
    assert lpf.update(0.4, 2.0) == 0.4  # nessun transitorio da zero
    y = lpf.update(0.8, 2.0)
    assert 0.4 < y < 0.8
    for _ in range(20):
        y = lpf.update(0.8, 2.0)
    assert y == pytest.approx(0.8, abs=1e-4)


# --- controllore PI (D5) ------------------------------------------------


def test_pi_convergenza_su_impianto_lineare():
    """Su un impianto y' = u - y il PI porta l'uscita al setpoint senza
    errore a regime (azione integrale)."""
    ctl = PIController(kp=0.5, ki=0.5, u0=0.0)
    y, target = 0.0, 0.6
    for _ in range(200):
        u = ctl.step(target - y, 0.1)
        y += 0.5 * (u - y) * 0.1
    assert y == pytest.approx(target, abs=1e-3)


def test_pi_anti_windup():
    """Setpoint irraggiungibile: l'integrale resta confinato in [lo, hi];
    quando l'errore inverte segno l'uscita stacca subito dal bordo."""
    ctl = PIController(kp=1.0, ki=1.0, u0=0.0)
    for _ in range(100):
        u = ctl.step(5.0, 1.0)  # errore enorme e costante
    assert u == 1.0
    assert ctl.integral == 1.0  # confinato, non 100+
    u = ctl.step(-0.2, 1.0)
    assert u < 1.0  # risposta immediata all'inversione


def test_pi_uscita_e_u0_nei_limiti():
    ctl = PIController(kp=10.0, ki=0.0, lo=0.0, hi=1.0, u0=0.5)
    assert ctl.step(1e6, 1.0) == 1.0
    assert ctl.step(-1e6, 1.0) == 0.0
    with pytest.raises(ValueError):
        PIController(1.0, 1.0, u0=2.0)


# --- integratore incrementale (D4) --------------------------------------


def test_surprise_rms_equivale_a_leaky_rms():
    rng = random.Random(3)
    scored = [
        ScoredEvent(t * 0.31, F(0), "onset", rng.random(), {})
        for t in range(50)
    ]
    batch = leaky_rms(scored, tau_s=3.0)
    inc = SurpriseRMS(3.0)
    for (t, v), ev in zip(batch, scored):
        assert inc.update(ev.time_s, ev.surprise) == pytest.approx(v)
    assert inc.value == pytest.approx(batch[-1][1])


def test_surprise_rms_ignora_dt_negativi():
    """Un'omissione emessa in ritardo dal cieco (tempo nel passato) non deve
    far esplodere l'EMA: Δt negativo trattato come simultaneo."""
    inc = SurpriseRMS(3.0)
    inc.update(10.0, 0.5)
    v = inc.update(9.0, 1.0)  # nel passato: alpha=1, nessun contributo
    assert v == pytest.approx(0.5)
    assert inc.update(11.0, 0.5) <= 1.0


# --- loop chiuso: contratto e determinismo (D9) --------------------------


def test_loop_deterministico_e_serializzabile():
    r1 = make_loop("oracle").run(30, SetpointCurve.constant(0.4), random.Random(5))
    r2 = make_loop("oracle").run(30, SetpointCurve.constant(0.4), random.Random(5))
    assert r1.trace == r2.trace and r1.events == r2.events

    assert len(r1.trace) == 30
    assert all(0.0 <= p.sigma <= 1.0 for p in r1.trace)
    assert all(p.surprise_rms >= 0.0 for p in r1.trace)
    data = json.loads(trace_to_json(r1.trace))
    assert len(data) == 30
    assert set(data[0]) == {"t", "target", "sigma", "surprise_rms"}


def test_loop_congelato_e_prologo():
    """controller=None: anello aperto a sigma0 costante; con controller,
    prima di engage_measure la sigma resta sigma0 (warm-lock, D5)."""
    gen = BernoulliGenerator(4, {2})
    frozen = ClosedLoop(gen, BPM, OracleListener(4, BPM), None, sigma0=0.7)
    res = frozen.run(10, SetpointCurve.constant(0.4), random.Random(1))
    assert all(p.sigma == 0.7 for p in res.trace)

    res = make_loop("oracle", engage=12).run(
        20, SetpointCurve.constant(0.4), random.Random(1)
    )
    assert all(p.sigma == 0.1 for p in res.trace[:12])
    assert any(p.sigma != 0.1 for p in res.trace[12:])


# --- firme dinamiche closed-loop (D10, D11-M4) ---------------------------


def test_omeostasi_oracolo_setpoint_costante():
    """Contro l'oracolo (riferimento fisso, nessuna assuefazione) il PI
    tiene il setpoint: errore a regime piccolo e sigma interna, mai in
    saturazione — il controllore trova il punto fisso (misurato ~0.02)."""
    for seed in (0, 6, 9):
        res = make_loop("oracle").run(
            80, SetpointCurve.constant(0.40), random.Random(seed)
        )
        late = res.trace[50:]
        err = statistics.mean(abs(p.surprise_rms - 0.40) for p in late)
        assert err < 0.05
        sig = [p.sigma for p in late]
        assert 0.05 < min(sig) and max(sig) < 0.95  # mai ai bordi
        assert statistics.pstdev(sig) < 0.2  # assestata, non in deriva


def test_inseguimento_rampa_oracolo():
    """Inseguimento di una rampa di setpoint (D5): la sorpresa segue il
    target filtrato con errore medio piccolo (misurato ~0.02)."""
    curve = SetpointCurve.ramp(0.38, 0.46, 40.0, 120.0)
    for seed in (1, 2, 3):
        loop = make_loop("oracle", engage=0, u0=0.3, sigma0=0.3)
        res = loop.run(80, curve, random.Random(seed))
        errs = [abs(p.surprise_rms - p.target) for p in res.trace[15:]]
        assert statistics.mean(errs) < 0.04
        assert max(errs) < 0.10
        pre = statistics.mean(p.surprise_rms for p in res.trace[15:20])
        post = statistics.mean(p.surprise_rms for p in res.trace[70:])
        assert post - pre > 0.05  # la salita c'è stata
        assert abs(post - 0.46) < 0.04  # e arriva al plateau


def test_ipotesi_centrale_la_leva_viene_assorbita():
    """Test closed-loop dell'ipotesi centrale (D10): lo stesso setpoint,
    raggiunto contro l'oracolo, è irraggiungibile contro il cieco.

    Il cieco, libero di scegliere il livello metrico, aggancia la
    suddivisione (tactus = ottavo): l'intera palette {2} gli resta
    on-grid e sigma smette di essere una leva di sorpresa (banda a regime
    ~0.21-0.24 contro il target 0.40). Il controllore satura a sigma=1
    senza mai avvicinarsi al target: l'assuefazione emerge
    dall'entrainment in forma ancora più forte del previsto — non
    "parametri in deriva" ma leva neutralizzata dalla scelta di livello.
    """
    target = 0.40
    for seed in (0, 6, 9):
        res = make_loop("blind").run(
            80, SetpointCurve.constant(target), random.Random(seed)
        )
        late = res.trace[50:]
        # saturazione permanente: nessun punto fisso interno
        assert all(p.sigma == 1.0 for p in late)
        # e il target resta lontano: la sorpresa vive al piano del cieco
        rms = statistics.mean(p.surprise_rms for p in late)
        assert rms < 0.30
        assert abs(rms - target) > 0.10


def test_ablazione_cieco_sotto_oracolo_a_sigma_statica():
    """D10, ablazione: a sigma=1.0 statica il cieco sta sistematicamente
    sotto l'oracolo — la divergenza è la misura dell'assuefazione (il
    cieco adotta il metro spostato o assorbe al livello suddivisione;
    l'oracolo, ancorato al metro notazionale, resta sorpreso)."""
    for seed in (0, 4, 7):
        gen = BernoulliGenerator(4, {2})
        seq = gen.sequence(60, random.Random(seed), density=1.0, syncopation=1.0)
        events = render(seq, bpm=BPM)
        oracle = leaky_rms(
            OracleListener(4, BPM).listen(events, n_measures=60), 3.0
        )
        blind = leaky_rms(
            BlindListener(4).listen(events, duration_s=120.0), 3.0
        )
        o_late = statistics.mean(v for t, v in oracle if t > 80)
        b_late = statistics.mean(v for t, v in blind if t > 80)
        assert b_late < o_late - 0.05
