# Interprete CTRNN a due neuroni con supervisore di active inference

Primo esperimento eseguibile: un agente SPM12 sceglie una prova per identificare una rete bersaglio sconosciuta, poi seleziona il codice con cui un **unico interprete CTRNN a pesi fissi** ne approssima la dinamica. Le reti bersaglio hanno due neuroni; l'interprete ne ha quattordici. Il catalogo comprende due programmi.

Il supervisore di active inference è esterno alla CTRNN: inferenza, valutazione delle politiche e selezione del programma sono eseguite da `spm_MDP_VB_X`. I neuroni dell'interprete eseguono il programma selezionato; non implementano ancora gli aggiornamenti di free energy.

## Esecuzione

Cartella prevista: `~/tools/CTRNN/active_inference_two_neuron`. Sono necessari MATLAB oppure GNU Octave, il repository CTRNN e SPM12 installato in `~/tools/spm12`. La dinamica è deterministica; il rumore del sensore è rappresentato nel modello osservativo e non richiede il toolbox Statistics.

In MATLAB o Octave:

```matlab
addpath(fullfile(getenv('HOME'), 'tools', 'CTRNN', 'active_inference_two_neuron'));
report = run_two_neuron_active_inference;
```

Per una sessione senza grafici:

```matlab
cfg = struct('make_plots', false);
report = run_two_neuron_active_inference(cfg);
```

Da shell, senza interfaccia grafica:

```bash
octave --no-gui --quiet --eval "addpath(fullfile(getenv('HOME'),'tools','CTRNN','active_inference_two_neuron')); report=run_two_neuron_active_inference(struct('make_plots',false));"
```

L'esperimento predefinito esegue 16 episodi per contesto e condizione, con seed `20260922`. Le condizioni sono tre, quindi gli episodi totali sono 96. Ogni episodio riparte dal proprio prior e non eredita apprendimento dall'episodio precedente. Lo stato del generatore casuale e il path vengono ripristinati alla fine.

I risultati vengono scritti in `results/`: `results.mat`, `summary.csv`, `trials.csv` e `two_neuron_summary.tex`; con i grafici attivi viene prodotto anche `two_neuron_demo.png`. Il frammento TeX è destinato agli appunti di ricerca. Per cambiare la destinazione o disattivare l'esportazione usare `cfg.output_dir` oppure `cfg.save_results=false`.

Percorsi alternativi:

```matlab
cfg = struct('ctrnn_root', '/percorso/CTRNN', ...
             'spm_root', '/percorso/spm12', 'make_plots', false);
report = run_two_neuron_active_inference(cfg);
```

Sono riconosciute anche le variabili d'ambiente `CTRNN_ROOT` e `SPM12_ROOT`.

## Reti, codice e interprete

Il codice originale simula stati nello spazio delle attività:

$$
\tau_i\dot s_i=-s_i+\sigma\!\left(\sum_j W_{ij}s_j+\sum_k E_{ik}u_k+b_i\right).
$$

La simulazione usa direttamente `createCTRNN`, `mulation`, `weights2program` e `runCTRNN` del repository, senza modificarli. Entrambe le reti hanno bias nulli, costanti di tempo pari a 5 e stato iniziale $(0.5,0.5)$:

| Bersaglio | $W$ ricorrente | $E$ degli ingressi | Codice |
| --- | --- | --- | --- |
| 1 | $\operatorname{diag}(2,-2)$ | $\operatorname{diag}(-2,2)$ | `[.7 .3 .3 .7]` |
| 2 | $\operatorname{diag}(-2,2)$ | $\operatorname{diag}(2,-2)$ | `[.3 .7 .7 .3]` |

I quattro valori del codice rappresentano `[W11 W22 E11 E22]`, trasformati linearmente dall'intervallo `[-5,5]` a `[0,1]`. Non sono bit. I collegamenti fuori diagonale rimangono nulli.

Quattro chiamate a `mulation` sostituiscono i quattro pesi programmabili con altrettanti moduli di tre neuroni: due neuroni iniziali più dodici neuroni ausiliari. Gli ingressi dell'interprete sono i due stimoli fisici e i quattro valori del programma; la lettura usa i primi due neuroni. La costruzione a partire da ciascun bersaglio produce matrici fisiche interne ed esterne **identiche**, verificate dal test. Cambia il codice fornito in ingresso. Il passo Euler predefinito è `dt=0.05`, la durata è 60 unità di tempo e la costante di tempo dei moduli è `0.25`.

Le prove disponibili sono lo stimolo neutro `(0.5,0.5)` e quello informativo `(0.9,0.5)`. Nel bersaglio, la prima attività a regime vale circa `0.5` in entrambi i contesti sotto la prova neutra, e `0.1969` oppure `0.6312` sotto quella informativa. La seconda attività resta `0.5` in queste prove. Le due reti bersaglio sono sistemi contraenti che convergono a punti fissi.

## Modello SPM12

Il modello discreto ha due fattori: identità del programma bersaglio e modalità dell'esperimento. Un episodio contiene tre tempi: stato iniziale, osservazione dopo la prova, esecuzione del programma scelto. Il feedback di corrispondenza arriva soltanto al tempo finale.

La tabella delle probabilità osservative è costruita **prima degli episodi** simulando ciascun programma sotto ciascuna prova. La misura usa la media del primo neurone nell'ultimo 10% del rollout. Un sensore gaussiano, con deviazione standard `0.08` e soglia `0.5`, produce due esiti, `low` e `high`. Le probabilità dei due intervalli sono calcolate con la CDF gaussiana tramite `erfc`; SPM campiona questi esiti discreti. Non si aggiunge rumore alle traiettorie CTRNN.

Si mantiene distinta la sorgente dei dati dal modello dell'agente:

- `mdp.A` descrive il **processo generativo** e deriva dalle risposte delle reti bersaglio dedicate.
- `mdp.a` codifica il **modello interno** e deriva dalle risposte dell'interprete programmato. Sono parametri di Dirichlet con concentrazione `1e8`; `eta=0` disattiva il loro apprendimento. L'alta concentrazione rende trascurabile l'incertezza sui parametri; resta l'incertezza sul contesto.

Viene chiamato il solver SPM originale. Nei risultati `negative_efe_initial` conserva la convenzione di SPM: valori maggiori della quantità restituita favoriscono la politica. La prova informativa ha un costo preferenziale di `0.12`; il feedback finale ha preferenze `+3` per la corrispondenza e `-3` per la mancata corrispondenza.

Il successo significa **selezionare l'identità del codice bersaglio**. La RMSE delle traiettorie è una misura separata di fedeltà dell'esecuzione: non è il reward dell'agente. I rollout delle combinazioni bersaglio/codice sono precalcolati e usati per attribuire l'errore all'esecuzione selezionata. Le credenze dopo la prova sono estratte prima del feedback terminale.

| Condizione | Conoscenza iniziale | Prove ammesse |
| --- | --- | --- |
| `active` | Prior uniforme sui due contesti | Neutra o informativa |
| `passive` | Prior uniforme sui due contesti | Solo neutra |
| `oracle` | Programma bersaglio noto | Neutra o informativa |

`passive` è un controllo con azioni limitate, **non** un'ablazione del termine epistemico del solver. `oracle` controlla se un agente che conosce già il programma evita una prova costosa. L'identità vera entra nel processo generativo, nelle metriche e nel prior dell'oracolo; non viene fornita al prior delle altre condizioni.

## Risultato della prima esecuzione

Esperimento e test completo eseguiti con GNU Octave 11.1.0, seed `20260922`, 16 episodi per contesto e condizione; tutti gli assert sono passati.

| Condizione | Codici corretti | Prove informative | Probabilità media del contesto vero dopo la prova | RMSE media di esecuzione |
| --- | --- | --- | --- | --- |
| `active` | 32/32 | 32/32 | `0.999345` | `0.021191` |
| `passive` | 15/32 | 0/32 | `0.488432` | `0.157719` |
| `oracle` | 32/32 | 0/32 | circa `1` | `0.021191` |

Sono risultati descrittivi di un singolo campione con seed fissato. Mostrano il comportamento previsto in questo esperimento; il confronto non fornisce una stima generale delle prestazioni. La compatibilità MATLAB è prevista dal codice, ma questa verifica è stata eseguita in Octave.

## Test e limiti della verifica

In MATLAB o Octave, dopo aver aggiunto questa cartella al path:

```matlab
report = test_two_neuron_active_inference;
```

Il test completo controlla fedeltà neurale, invariabilità dei pesi, differenza fra codici corretti e sbagliati, ricerca d'informazione, credenze prima del feedback e controlli `passive`/`oracle`. Verifica inoltre che prior e politiche iniziali non dipendano dal contesto nascosto, e che il modello `a` non venga appreso. Per i soli controlli numerici CTRNN:

```matlab
backend = ai2_test_backend;
```

Il controllo del solo backend, eseguito in Octave con i parametri predefiniti, ha dato:

| Controllo | Risultato |
| --- | --- |
| RMSE massima sulle quattro combinazioni contesto/prova | `0.027485` |
| RMSE su sequenza di ingressi di verifica, codici corretti | `0.025316 / 0.025531` |
| RMSE sulla stessa sequenza, codici sbagliati | `0.156321 / 0.158105` |
| Variazione massima delle previsioni a regime dimezzando `dt` | `0.000127` |

La sequenza di verifica contiene quattro blocchi di 15 unità di tempo: `(0.5,0.5)`, `(0.9,0.5)`, `(0.2,0.8)` e `(0.7,0.3)`. Non è usata per costruire la tabella osservativa del supervisore. Il compito SPM, invece, usa le stesse prove e lo stesso catalogo impiegati nella calibrazione: verifica identificazione attiva entro questo compito, senza misurare generalizzazione a nuove reti.

I moltiplicatori originali sono approssimati. Sotto la prova neutra l'interprete prevede circa `0.5359` e `0.5070` per il primo neurone, mentre entrambi i bersagli producono `0.5`. Questa piccola informazione apparente sul contesto è un errore del modello; resta presente nell'esperimento. La scelta del programma corretto non elimina l'errore di emulazione.

Il prototipo riguarda una famiglia finita con topologia, bias e costanti di tempo fissati. Non dimostra universalità per CTRNN arbitrarie, emulazione di cicli limite, inferenza continua dei pesi o cambiamento del programma durante un rollout. Una prossima estensione è sostituire il catalogo discreto con una distribuzione sul codice e misurare esplicitamente l'effetto dell'errore dell'interprete sull'inferenza.

## Cache temporanea e file

Per evitare la latenza di scansione dei path su cartelle sincronizzate con OneDrive, l'esecuzione principale e il test completo creano per default una cache in una directory temporanea. `ai2_setup.m` copia in modalità binaria i soli sorgenti necessari **dalle installazioni locali** di CTRNN e `~/tools/spm12`; non riscrive né sostituisce gli algoritmi. L'elenco esatto è in `ai2_setup.m`. Le copie sono rimosse a fine esecuzione e i percorsi originali sono conservati nel report. I percorsi della cache presenti nel report indicano quindi file che non persistono dopo il ritorno della funzione.

Per usare direttamente le installazioni, eventualmente con maggiore latenza:

```matlab
report = run_two_neuron_active_inference(struct('use_runtime_cache', false));
```

Il solo `ai2_test_backend` usa direttamente CTRNN; il test completo gli prepara invece la cache. Non usare `genpath` sull'intero repository: alcune cartelle storiche contengono funzioni omonime, ad esempio `sigmoid.m`.

| File | Ruolo |
| --- | --- |
| `run_two_neuron_active_inference.m` | Esperimento e raccolta dei risultati |
| `ai2_build_bank.m` | Bersagli, interprete e calibrazione delle risposte |
| `ai2_simulate.m` | Rollout attraverso l'integratore originale |
| `ai2_make_mdp.m` | Processo, modello osservativo, politiche e preferenze SPM |
| `ai2_setup.m` | Dipendenze e cache temporanea dei sorgenti |
| `ai2_write_results.m` | Esportazione MAT, CSV e frammento TeX |
| `ai2_plot_results.m` | Figura delle dinamiche e dei risultati |
| `ai2_test_backend.m` | Fedeltà numerica e verifiche sugli ingressi |
| `test_two_neuron_active_inference.m` | Verifiche complete di comportamento e assenza di informazioni anticipate |
