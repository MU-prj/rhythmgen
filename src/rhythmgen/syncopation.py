"""Indici di sincopazione di riferimento per l'ancoraggio esterno (design D10).

Implementazioni simboliche sulla griglia binaria di 16 sedicesimi in 4/4,
usate per validare la valuta di sorpresa dell'oracolo (M2). Questi indici
assumono un metro DATO: vanno confrontati con l'ascoltatore oracolo (o con la
finestra iniziale del cieco), mai col cieco a regime (D10).
"""

from typing import Iterable

# Pesi metrici della griglia binaria 4/4 a sedicesimi (albero metrico LHL).
LHL_WEIGHTS_16 = [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1]


def lhl_syncopation(onsets: Iterable[int]) -> int:
    """Longuet-Higgins & Lee (1984), variante di Fitch & Rosenfeld (2007).

    Per ogni posizione vuota r preceduta (circolarmente) dall'ultimo onset n
    con peso metrico minore, somma w(r) − w(n). Zero per pattern isocroni
    sui livelli forti (four-on-the-floor, corsa di ottavi).
    """
    filled = set(onsets)
    if not filled:
        return 0
    n = len(LHL_WEIGHTS_16)
    score = 0
    for r in range(n):
        if r in filled:
            continue
        d = 1
        while (r - d) % n not in filled:
            d += 1
        prev = (r - d) % n
        if LHL_WEIGHTS_16[r] > LHL_WEIGHTS_16[prev]:
            score += LHL_WEIGHTS_16[r] - LHL_WEIGHTS_16[prev]
    return score


def toussaint_complexity(onsets: Iterable[int]) -> int:
    """Complessità metrica di Toussaint (The Geometry of Musical Rhythm).

    (Somma dei k pesi massimi disponibili sulla griglia) − (somma dei pesi
    delle posizioni occupate): quanto gli onset evitano i punti forti,
    a parità di numero di onset. Ignora le pause per costruzione.
    """
    filled = set(onsets)
    best = sum(sorted(LHL_WEIGHTS_16, reverse=True)[: len(filled)])
    return best - sum(LHL_WEIGHTS_16[i] for i in filled)
