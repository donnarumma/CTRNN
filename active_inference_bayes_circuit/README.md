# B1 — Inferenza bayesiana in un interprete CTRNN programmabile

Questo primo esperimento dell’**ipotesi B** fa eseguire alla rete un’inferenza percettiva binaria. Il programma specifica un modello osservativo; l’ingresso specifica l’osservazione; le attività di due neuroni approssimano il posteriore. Una sola rete interprete di **14 neuroni**, con pesi fisici fissi, esegue i diversi modelli cambiando soltanto quattro ingressi di programma.

Nell’esperimento A, nella cartella sorella `active_inference_two_neuron`, SPM seleziona il programma. Qui **il calcolo neurale del posteriore avviene nella CTRNN**. SPM è soltanto un riferimento numerico indipendente e non alimenta il circuito. Anche se entrambi gli interpreti hanno 14 neuroni, i collegamenti resi programmabili sono diversi.

## Esecuzione

In MATLAB o Octave, dalla cartella `~/tools/CTRNN/active_inference_bayes_circuit`:

```matlab
report = run_bayesian_circuit;
checks = test_bayesian_circuit;
```

Le dipendenze sono il repository CTRNN e SPM12 già installati, cercati per default in `~/tools/CTRNN` e `~/tools/spm12`. Per percorsi differenti:

```matlab
cfg = struct('ctrnn_root', '/percorso/CTRNN', ...
             'spm_root', '/percorso/spm12', ...
             'output_dir', '/percorso/risultati');
report = run_bayesian_circuit(cfg);
```

Sono supportate anche le variabili d’ambiente `CTRNN_ROOT` e `SPM12_ROOT`. Per eseguire senza grafici o senza salvare file:

```matlab
report = run_bayesian_circuit(struct('make_plots', false, ...
                                    'save_results', false));
```

Il percorso MATLAB/Octave e lo stato del generatore casuale vengono ripristinati alla fine. L’esperimento neurale è deterministico.

## Modello e circuito

Le due ipotesi nascoste sono `H1` e `H2`, con prior fisso `[0.5, 0.5]`. Ogni modello è specificato da:

```text
[p1, p2] = [P(high | H1), P(high | H2)]
```

L’osservazione è un ingresso a due canali: `[1,0]` per `high`, `[0,1]` per `low`. I quattro esempi principali sono:

| Nome | `[p1,p2]` | `P(H1 | high)` esatto | `P(H1 | low)` esatto |
|---|---|---:|---:|
| strong | `[.8,.2]` | .8 | .2 |
| weak | `[.6,.4]` | .6 | .4 |
| reversed | `[.2,.8]` | .2 | .8 |
| asymmetric | `[.75,.4]` | .652174 | .294118 |

Il modello viene compilato in un circuito bersaglio di due neuroni, senza collegamenti ricorrenti tra i due output (`W = 0`), bias nulli e matrice degli ingressi:

```text
v = [log(p1/p2), log((1-p1)/(1-p2))]
V = [v; -v]
```

Per un’osservazione fissa, `L = v * osservazione` è il log-rapporto delle likelihood. Con `sigma` logistica:

```text
tau * dq1/dt = -q1 + sigma(L)
tau * dq2/dt = -q2 + sigma(-L)
```

A regime il circuito ideale restituisce le due probabilità posteriori. Il transitorio esegue il rilassamento neurale. Con uguali costanti temporali e inizializzazione `[.5,.5]`, gli output ideali rimangono complementari.

`mulation` sostituisce ciascuno dei quattro collegamenti esterni con il motivo moltiplicatore originale di tre neuroni: `2 + 4 × 3 = 14`. L’interprete ha due ingressi osservativi e quattro ingressi di programma, nell’ordine **`[E11 E12 E21 E22]`**. Il codice è la trasformazione affine dei quattro pesi nell’intervallo scelto; con `[-5,5]`, `programma = (pesi + 5)/10`.

Preparare i log-rapporti dal modello è una fase di compilazione. Il programma non contiene l’osservazione del singolo episodio né un posteriore già calcolato. Durante ogni simulazione, il circuito riceve soltanto osservazione e programma; Bayes analitico e SPM servono per misurare il risultato.

## Che cosa verifica l’esperimento

Il run predefinito usa i quattro esempi e una griglia di 25 coppie con `p1,p2 ∈ {.2,.35,.5,.65,.8}`. Sono **58 episodi**, due osservazioni per ciascuna delle 29 configurazioni; due esempi coincidono con punti della griglia, quindi i modelli distinti sono 27. Ogni episodio riparte dalle stesse condizioni iniziali. Non c’è addestramento.

| Parametro | Default |
|---|---:|
| Passo di integrazione Euler | `.05` |
| Durata | `60` |
| Costante temporale dei neuroni bersaglio | `5` |
| Costante temporale dei moltiplicatori | `.25` |
| Intervallo dei pesi codificati | `[-5,5]` |

La verifica confronta il circuito bersaglio con Bayes analitico, l’interprete con entrambi e i posteriori con il solver originale `spm_MDP_VB_X`. Controlla inoltre che matrici, bias, costanti temporali, condizioni iniziali, dimensioni e funzione di attivazione dell’interprete siano identici fra i programmi.

SPM viene usato in modalità HMM, con una sola osservazione (`T=1`), senza azioni o politiche. La sua costante numerica `tau=1` consente la convergenza dell’aggiornamento statico; non è la costante temporale della CTRNN. Il confronto SPM avviene dopo le simulazioni principali e non ne determina gli output.

Sui quattro esempi vengono eseguiti anche controlli con passo dimezzato e con intervallo `[-2,2]`. Quest’ultimo cambia la famiglia rappresentabile e la codifica: ridurre l’intervallo non garantisce di migliorare tutte le misure dell’approssimazione.

## Free energy e misure

Per il circuito ideale, indicando con `p*` il posteriore esatto e con `q` il primo output:

```text
F(q) = KL([q,1-q] || [p*,1-p*]) - log P(osservazione)
dF/dt = (p* - q)/tau × [logit(q) - logit(p*)] ≤ 0
```

La discesa vale per un’osservazione fissa; con `0 < dt/tau ≤ 1`, anche il passo Euler ideale riduce F per convessità. È una proprietà del circuito bersaglio, non una garanzia automaticamente trasferita ai moltiplicatori approssimati.

Le misure distinguono:

- **Errore grezzo:** massimo scarto fra i due output finali effettivi e il posteriore esatto.
- **Difetto di normalizzazione:** massimo di `abs(q1+q2-1)` durante la traiettoria.
- **KL e free energy diagnostiche:** calcolate normalizzando esternamente la coppia di output; questa normalizzazione non è eseguita dal circuito e non viene reinserita negli ingressi.
- **Transitori:** errore di emulazione, variazione finale degli output e aumenti di free energy, riportati anche quando il risultato finale è vicino a Bayes.

**Verifica completata in Octave 11.1.0: tutti i test passano.** Con la stessa osservazione `high`, i programmi producono questi output finali:

| Modello | Bayes `[q1,q2]` | Output grezzi CTRNN `[r1,r2]` |
|---|---|---|
| strong | `[.800000,.200000]` | `[.786256,.243949]` |
| weak | `[.600000,.400000]` | `[.611874,.433943]` |
| reversed | `[.200000,.800000]` | `[.243949,.786256]` |
| asymmetric | `[.652174,.347826]` | `[.653905,.390225]` |

Sui 58 episodi, il massimo errore grezzo finale è **0.046853**, il massimo difetto di normalizzazione è **0.047878** e la massima KL diagnostica finale è **0.004054 nat**. Il circuito bersaglio ideale raggiunge Bayes con errore massimo `1.735e-6`; lo scarto SPM è `2.22e-16`.

Il circuito ideale non mostra aumenti di free energy. L’interprete mostra invece incrementi transitori: il massimo è **0.000798 nat per passo**. Non possiamo quindi attribuire all’interprete approssimato la monotonia dimostrata per il bersaglio ideale.

Un caso utile per leggere le misure è il modello non informativo `p1=p2`: gli output grezzi sono `[.52388497,.52388497]`, mentre Bayes restituisce `[.5,.5]`. La KL diagnostica è zero dopo normalizzazione esterna, ma l’errore neurale grezzo e il difetto di complementarità restano presenti.

Sui quattro esempi, dimezzare il passo cambia gli output finali al massimo di `5.63e-8`. Usare il range `[-2,2]` produce un massimo errore grezzo di **0.049570** sugli stessi esempi, contro **0.043949** nel default: restringere il range non migliora uniformemente l’approssimazione. Disattivando il riferimento SPM, le otto traiettorie principali rimangono identiche, campione per campione.

I risultati vengono salvati in `results/`: `results.mat` contiene il report completo; `posteriors.csv`, `summary.csv` e `programs.csv` contengono output, misure e programmi; `bayes_circuit_demo.png` mostra i confronti; `bayes_circuit_summary.tex` riporta i valori negli appunti. Le soglie dei test numerici caratterizzano questa famiglia e questa configurazione; non costituiscono limiti teorici dell’errore.

## Codice e provenienza

| File | Funzione |
|---|---|
| `run_bayesian_circuit.m` | Esegue episodi, confronti e sensibilità |
| `bci_build_circuits.m` | Compila modelli e verifica l’identità dell’hardware |
| `bci_simulate.m` | Esegue la dinamica tramite `runCTRNN` originale |
| `bci_measure.m` | Calcola riferimenti e diagnostica, fuori dal circuito |
| `bci_spm_reference.m` | Confronto indipendente con SPM12 |
| `bci_setup.m` | Risolve dipendenze e prepara il percorso temporaneo |
| `test_bayesian_circuit.m` | Verifica numerica riproducibile |

Per evitare scansioni lente delle cartelle sincronizzate, il default usa copie temporanee **identiche byte per byte** di 8 file CTRNN e 13 file SPM letti dalle installazioni locali. Non sono riscritture dei solver, non sono aggiunte al repository e vengono eliminate alla fine. `report.paths` conserva i percorsi delle sorgenti e del runtime. Impostare `use_runtime_cache=false` usa direttamente le cartelle installate. Le sorgenti originali CTRNN e SPM rimangono inalterate.

## Portata del risultato

B1 dimostra il nucleo percettivo dell’ipotesi B: **un modello fornito come programma configura un circuito a pesi fissi che approssima un aggiornamento bayesiano**. I due neuroni bersaglio non eseguono ancora message passing ricorrente; la ricorrenza aggiunta appartiene ai motivi moltiplicatori.

Il prior resta uniforme, ogni episodio contiene una sola osservazione e viene resettato. Non sono implementati accumulo di evidenza, apprendimento del programma, scelta di azioni, expected free energy o active inference completa. Il successivo passo sperimentale è aggiungere memoria neurale per aggiornare il prior con evidenza successiva, mantenendo espliciti gli errori di emulazione.
