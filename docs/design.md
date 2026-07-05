# rhythmgen — Design

Documento di design consolidato dalla sessione di grilling del 2026-07-05.
Ogni sezione registra una decisione, la sua motivazione e le alternative scartate.

## Idea

Sistema generativo di ritmi fondato sulla Dynamic Attending Theory (Jones;
Large & Jones 1999): un generatore probabilistico di onset e un ascoltatore
interno adattivo, chiusi in un feedback loop. L'ascoltatore misura la sorpresa
degli eventi rispetto alle proprie aspettative; un controllore omeostatico
modula i parametri del generatore per mantenere la sorpresa vicino a un
setpoint.

**Ipotesi centrale (falsificabile).** Contro un ascoltatore adattivo che
inferisce il metro dai soli onset, nessuna configurazione statica di parametri
sostiene la sorpresa: l'assuefazione emerge gratuitamente dall'entrainment
(un profilo sincopato costante viene ri-agganciato come nuovo metro). Il
controllore, per mantenere il setpoint, è quindi costretto a muovere
continuamente i parametri. Se questo accade, il ritmo vive nella
non-stazionarietà; se il controllore trova un punto fisso, l'ipotesi è
falsificata. In entrambi i casi c'è un risultato.

## Decisioni

### D1 — Deliverable: modello di ricerca offline

Libreria Python che genera sequenze simboliche e le rende in timestamp, non in
tempo reale. Il loop gira in simulazione: determinismo (seed), test, grafici
degli indici. Il real-time è un'estensione futura, non un vincolo di progetto.

### D2 — Ascoltatore a due modalità selezionabili

L'ascoltatore vive dietro un'interfaccia unica con due implementazioni che
condividono lo stesso contratto di uscita (la valuta di aspettativa, D4):

- **Oracolo**: conosce griglia e tempo (li eredita dai dati di partenza).
  Nessuna inferenza; fase metrica esatta. Serve da baseline, da strumento di
  debug e da ancora per la validazione esterna.
- **Cieco**: riceve solo timestamp (ed eventualmente velocity) e inferisce
  periodo e fase con oscillatori adattivi. Può sbagliare e perdere il beat:
  è il regime scientificamente interessante.

La differenza di sorpresa tra cieco e oracolo non è rumore da tollerare: è il
segnale di difficoltà inferenziale della sequenza.

### D3 — Ascoltatore cieco: banco gerarchico di oscillatori

2–3 oscillatori accoppiati (tactus, suddivisione, misura) alla Large & Jones:
ogni livello ha fase φ, periodo τ, un pulse attenzionale (coseno rialzato o
simile) e costanti di adattamento proprie — la misura si adatta più lentamente
della suddivisione. La sincope emerge dalla doppia contabilità: evento previsto
dal livello suddivisione ma in punto di bassa aspettativa del livello misura.
Un oscillatore singolo è sordo alle sincopi e invaliderebbe D2.

### D4 — Valuta unica della sorpresa: l'aspettativa

Ogni livello espone E(φ) ∈ [0,1] (il pulse attenzionale). Sorpresa di un
onset = 1 − E(φ) al momento del colpo; sorpresa di un'omissione = E al picco
quando il picco passa senza evento entro una finestra di aggancio. L'errore di
fase in millisecondi non appare mai come numero: è codificato nella forma del
pulse, quindi la valuta è adimensionale e indipendente dal BPM. Combinazione
tra livelli: somma pesata Σ wᵢ·Sᵢ; integrazione con media mobile RMS / leaky
integrator su finestra di 2–4 s.

Vincolo: i picchi del livello misura NON vanno normalizzati tutti a 1.
L'ampiezza del picco per posizione metrica è informazione: saltare il battere
deve sorprendere più che saltare un levare.

Scartato: somma pesata di grandezze eterogenee (ms + eventi binari + derive di
periodo) — pesi arbitrari, invalidati da ogni ritaratura del tempo.

### D5 — Macrostruttura = traiettoria di setpoint

Fase 1: controllore omeostatico con setpoint costante (parametro utente).
Le macro-sezioni e la "condizione di arrivo" si esprimono poi come una curva
S_target(t) in input (costante, rampa, spezzata): un solo meccanismo copre
statico, deriva e arco narrativo. Controllore PI con anti-windup, filtro
passa-basso sul setpoint, risposta su scale musicali (secondi, non ms);
l'ascoltatore si adatta più in fretta di quanto il controllore muova il
generatore.

L'ibrido a macro-sezioni esplicite (macchina a stati con transizioni innescate
da indicatori a lungo termine) è tracciato come evoluzione futura nella
[issue #1](https://github.com/MU-prj/rhythmgen/issues/1).

### D6 — Griglia: razionali esatti, palette piatta

Posizioni come `fractions.Fraction`, palette dichiarata a monte come insieme di
suddivisioni attive (es. sedicesimi + terzine di ottavi); l'unione genera le
posizioni ammesse. Confronti esatti, nessun drift float, serializzabile.

Convenzione di unità (fissata qui una volta per tutte): **le posizioni sono
frazioni di beat**, con il beat = 1; la misura è un numero intero di beat noto
al generatore. Terzina di ottavi → {0, 1/3, 2/3} dentro il beat.

Annidamenti arbitrari (terzina dentro terzina) esclusi in fase 1; estendibili
in futuro come denominatori compositi mantenendo la struttura piatta.

### D7 — Generatore: campo di Bernoulli parametrico

Ogni posizione ha un peso metrico a priori w(pos) dalla gerarchia notazionale.
Probabilità di evento p(pos) = f(density, syncopation, tuplet_mix, w(pos)):

- `syncopation` σ: w′ = (1−σ)·w + σ·(1−w). Leva primaria del controllore.
- `density`: scala la massa totale. Normalizzata sul **numero atteso di eventi
  per misura** (Σp = density·N_beat), non sul massimo, così il significato non
  dipende dalla cardinalità della palette. Fissa o secondaria in fase 1.
- `tuplet_mix`: probabilità relativa delle posizioni irregolari.

Campionamento Bernoulli indipendente per posizione, nessuna memoria tra misure
in fase 1 (isolamento del loop di base; un parametro `continuity` markoviano è
un'estensione compatibile). Velocity/accenti dallo stesso profilo.

Avvertenze note:
- L'effetto di density sulla sorpresa è a campana (massimo a densità
  intermedie), non monotono.
- Anche σ non è monotono a regime: a σ=0.5 il profilo è piatto (massima
  entropia); a σ=1 il profilo invertito è regolare quanto l'originale e il
  cieco vi si ri-aggancia. Vedi ipotesi centrale.

Scartati: catene di Markov su pattern (manopole discrete, relazione
parametro→sorpresa opaca per un PI), grammatiche (spazio parametri non
continuo).

### D8 — Rendering: solo jitter gaussiano in fase 1, default 0

Conversione deterministica Fraction → secondi (tempo base costante) più un
unico parametro σ_t di jitter gaussiano i.i.d. sugli onset, default 0.
Isola la sorpresa metrica negli esperimenti; σ_t serve a testare robustezza
dell'aggancio e taratura della larghezza del pulse. Swing e groove sistematici
(offset stazionari per posizione) sono fase 2: un ascoltatore adattivo li
assorbe nell'aggancio, quindi non contribuiscono alla dinamica a regime.

Terminologia: il jitter i.i.d. è statisticamente *stazionario*; la sua
proprietà rilevante è che è **inapprendibile** (non predicibile campione per
campione), a differenza dello swing.

### D9 — Stack: core stdlib puro + extras

Core (griglia, generatore, ascoltatori, controllore) solo stdlib: `fractions`,
`dataclasses`, `random`, `math`, `json`. Simulazione event-based: numpy non
compra nulla e nasconde la logica. Output serializzabile in JSON:

- eventi: `{onset_s, position (frazione), velocity}`
- traccia: `{t, surprise_rms, phase_coherence}` per livello

Extras opzionali fuori dal core: `analysis` (numpy/matplotlib per gli
esperimenti), `preview` (click-track WAV via modulo stdlib `wave`), export
MIDI (`mido`) solo quando servirà.

### D10 — Valutazione: firme dinamiche + ancoraggio esterno

Test d'ascolto umani rimandati a valle (eventuale fase finale di tesi).

**Firme dinamiche falsificabili** (script riproducibili, seed fissi):
- Open-loop: sweep dei parametri → curve di sorpresa **per finestra
  temporale**, non su media globale: la media mescola transitorio e regime e
  appiattisce tutto. Attese: monotonia in σ nella finestra iniziale; campana o
  decadimento a regime.
- Closed-loop: errore di inseguimento del setpoint; test dell'ipotesi centrale
  (parametri congelati → sorpresa decade; controllore attivo → sorpresa al
  target e parametri in deriva).
- Ablazione cieco vs oracolo: divergenza nel tempo come misura della
  dipendenza della sincopazione dal riferimento.

**Ancoraggio esterno**: correlazione (Spearman) della sorpresa con gli indici
di Longuet-Higgins & Lee e di Toussaint su pattern noti (clave son, tresillo,
backbeat, bossa). Vincolo: correlare con l'**oracolo** (o la finestra iniziale
del cieco) — quegli indici assumono un metro dato; il cieco a regime correla
sempre peggio per costruzione, e quella divergenza è un risultato, non un
fallimento. Test con soglie stocastiche: seed fissi e molte ripetizioni.

### D11 — Roadmap: oracolo prima del cieco

Ogni milestone lascia un artefatto validabile in isolamento.

- **M1 — Griglia + generatore + rendering.** Test statistici open-loop:
  frequenze empiriche vs pesi, densità media = parametro, nessun float spurio.
- **M2 — Ascoltatore oracolo + valuta di sorpresa.** Fase metrica nota, nessun
  PLL. Validazione contro LHL/Toussaint: ogni discrepanza è imputabile alla
  valuta (forma del pulse, pesi dei livelli), non all'entrainment.
- **M3 — Ascoltatore cieco (banco di oscillatori).** Test di convergenza verso
  l'oracolo **solo nel regime facile** (σ bassa, jitter nullo): lì la
  divergenza è un bug. Ad alta sincopazione la divergenza è attesa
  (ri-entrainment sul metro spostato) ed è il dato, non il difetto.
- **M4 — Controllore PI + firme dinamiche closed-loop.** Omeostasi a setpoint
  costante, inseguimento di rampe, test dell'ipotesi centrale.

## Questioni aperte

- Forma esatta del pulse (coseno rialzato vs von Mises) e larghezze per
  livello: da tarare in M2 sull'ancoraggio esterno.
- Meccanismo di accoppiamento tra oscillatori del banco (reset di fase vs
  trascinamento morbido): da decidere in M3 sui dati di convergenza.
- Pesi wᵢ dei livelli nella somma della sorpresa: fissi in fase 1, adattivi
  (es. pesati dalla coerenza del livello) come estensione.
- Bootstrap del cieco: inizializzazione di τ (es. istogramma degli
  inter-onset-interval iniziali) — dettaglio di M3.

## Riferimenti di partenza

- M. R. Jones, *Time, our lost dimension* (1976); Large & Jones,
  *The dynamics of attending* (Psych. Review, 1999).
- E. W. Large, modelli di entrainment con oscillatori adattivi.
- Longuet-Higgins & Lee (1984), misura di sincopazione; G. Toussaint,
  *The geometry of musical rhythm*.
- M. Pearce, IDyOM (per contrasto: sorpresa information-theoretic senza
  oscillatori — alternativa scartata in D3 ma utile in discussione di tesi).
