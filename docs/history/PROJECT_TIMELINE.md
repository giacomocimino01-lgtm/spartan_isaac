# Timeline tecnica del progetto SPARTAN / dVRK Peg & Ring

Ultimo aggiornamento: **20 ottobre 2026**
Repository locale analizzata: `spartan_isaac`  
Repository di esecuzione sul server: `/home/aiprah/Documents/m_dVrk`

## Come leggere questo documento

Questa non è una cronologia ricavata soltanto dalla memoria o dai messaggi di commit. È una ricostruzione basata su:

- cronologia Git di tutti i branch;
- date e contenuti di log, checkpoint, dataset, report e file di stato presenti sul server;
- metriche CSV delle valutazioni deterministiche;
- configurazioni e script delle pipeline sperimentali.

Ogni evento usa uno dei seguenti livelli di confidenza:

- **[G] Git**: data e cambiamento verificati da un commit;
- **[A] Artifact**: evento verificato da log, report, checkpoint, dataset o risultati di evaluation;
- **[R] Ricostruzione**: interpretazione tecnica supportata da più evidenze, ma non descritta esplicitamente in un singolo commit o log.

Le date indicano quando esiste una prova dell'evento, non necessariamente il primo minuto in cui l'idea è stata concepita. Da settembre in poi molto lavoro è ancora non committato: in quel periodo gli artifact sul server sono una fonte più completa di Git.

---

## 1. Obiettivo e architettura emersi nel tempo

Il progetto simula con Isaac Lab una variante dell'esercizio chirurgico **Peg & Ring** eseguita da un dVRK. L'architettura si è progressivamente stabilizzata nella seguente forma:

1. una policy RL agisce come **reasoner di alto livello**;
2. le azioni semantiche della policy comandano due bracci attraverso la **SPARTAN State Machine**;
3. la State Machine traduce comandi come `reach`, `grasp`, `release` e `idle` in controllo low-level;
4. un encoder visivo TCC/XIRL codifica il frame corrente;
5. la distanza dall'embedding goal della fase fornisce la reward densa;
6. BC inizializza la policy da dimostrazioni valide;
7. PPO affina la strategia usando reward TCC, bonus terminale e, negli esperimenti più recenti, replay supervisionato;
8. un phase router seleziona l'encoder/policy della fase successiva per comporre il task completo.

Le cinque fasi operative sono:

| Fase | Stato iniziale tipico | Obiettivo |
|---|---|---|
| Phase 0 | anelli sui pioli iniziali, fissi o randomizzati | tutti e quattro gli anelli sul piolo verde |
| Phase 1 | tutti e quattro gli anelli sul verde | due anelli sul rosso e due sul blu |
| Phase 2 | due anelli sul rosso e due sul blu | tutti e quattro gli anelli rilasciati al centro della board |
| Phase 3 | tutti e quattro gli anelli al centro della board | esattamente un anello qualsiasi su ciascuno dei quattro pioli colorati |
| Phase 4 | un anello qualsiasi su ciascun piolo colorato | due anelli fuori dalla board a destra e due a sinistra |

Il goal TCC di una fase viene costruito dalla media degli embedding dei frame finali delle dimostrazioni. Per il prototipo di phase routing, lo start prototype è stato poi ridotto al **solo primo frame**, non alla media dei primi cinque frame.

---

## 2. Timeline sintetica

### Marzo–aprile 2026 — ambiente, asset e State Machine

**17 marzo — nascita della repository. [G]**  
Il commit `eb4064f` crea lo scheletro dell'estensione Isaac Lab: configurazione dell'ambiente, termini di reward, script random/zero-agent/RSL-RL e prima documentazione. È il primo punto verificabile del progetto.

**24–27 aprile — introduzione dello state-machine runner e della scena dVRK. [G]**  
I commit `8bf7087` e `91d87c7` aggiungono `run_state_machine.py`, configurazione del PSM, asset della board e una prima versione estesa del runner. Il controllo del task inizia a essere espresso tramite primitive semantiche, non direttamente come azioni articolari.

**28 aprile — sincronizzazione tra Isaac Lab e State Machine. [G]**  
Il commit `28074a8` migra le terminazioni nell'ambiente Isaac Lab, corregge percorsi/reset e introduce `parallel_run.py`. Questa è la base della futura esecuzione vettorializzata PPO.

> **Limite della ricostruzione:** tra marzo e maggio i commit sono pochi e molto grandi. Possiamo identificare con certezza i risultati consolidati nei commit, ma non l'ordine di ogni prova intermedia.

### Giugno 2026 — HFSM, raccolta dati TCC e valutazione

**5 giugno — prima pipeline sperimentale completa. [G]**  
Il commit `3dfd0dc` introduce o consolida:

- una HFSM/SPARTAN State Machine riusabile;
- `collect_tcc_dataset.py` per raccogliere dimostrazioni video;
- `eval_run.py` per evaluation;
- generazione di video;
- strumenti di analisi e plotting;
- `parallel_env.py` ed eventi dell'ambiente.

Da questo punto il progetto non è più solo un ambiente simulato: possiede il ciclo **dimostrazioni → reward visiva → training → evaluation/video**.

### Luglio 2026 — Phase 0 completata e arrivo della Behavioural Cloning

**29 luglio — primo grande milestone funzionale. [G]**  
Il commit `fab2d32` è esplicito: *“Finita la fase zero, training fase 1”*. Vengono aggiunti `collect_ppo_demos.py`, `pretrain_ppo.py` e `check_bc_policy.py`.

La pipeline diventa quindi:

```text
dimostrazioni della State Machine
        ↓
dataset TCC → encoder visivo / reward densa
        ↓
dataset di decisioni → Behavioural Cloning
        ↓
checkpoint BC → PPO
        ↓
evaluation deterministica
```

**[R]** La Phase 0 è considerata risolta nella configurazione dell'epoca; la Phase 1 è il nuovo obiettivo. Le prove successive mostreranno però che “risolta” in configurazione fissa/guidata non equivale ancora a robustezza sotto randomizzazione o maschera minimale.

### Agosto 2026 — riordino e riproducibilità della repository

**6 agosto — centralizzazione degli artifact. [G]**  
I commit `248d653`, `9e06c24` e il merge `f02c9f9` centralizzano configurazioni e output sotto `configs/` e `artifacts/`, estraggono moduli `controllers/` e `hrl/`, introducono metadata per i dataset e una CI.

Questo refactor è importante storicamente: separa l'algoritmo dalla disposizione casuale dei file e rende confrontabili checkpoint, report e dataset successivi.

### 1–4 settembre 2026 — robustezza e riproducibilità di Phase 0

**1 settembre — revisione Phase 0. [G]**  
Il commit `15a6f46` riapre il lavoro sulla Phase 0, modificando reset ed evaluation.

**2 settembre — controlli deterministici PPO. [G]**  
Il commit `47baad9` aggiunge ranking dei checkpoint, policy mascherata, schedule, salvataggio controllato e test. Da qui la selezione non dipende più solo dall'ultimo checkpoint o dal successo mostrato durante il training.

**3 settembre — correzioni semantiche e fisiche. [G]**

- `983b023` ripristina il bonus esplicito di successo logico;
- `240a71e` limita le reach bloccate e applica controlli di raggiungibilità per braccio.

Queste modifiche riducono gli episodi da 600 step causati da comandi impossibili, timeout e `last_target` incoerenti. Il bonus terminale torna a distinguere esplicitamente il completamento del task, mentre TCC continua a fornire la reward densa.

**4 settembre — curriculum di randomizzazione. [G][A]**  
I commit `53f20ac`, `179bb06`, `9698f8b` e `4708c2d` introducono randomizzazione progressiva degli anelli e avanzamento del curriculum legato al successo. `eb5118c` assegna nomi unici ai checkpoint finali.

Le evaluation deterministiche mostrano il trade-off:

- configurazione fissa con reach guard: **20/20 = 100%** per tutti i quattro checkpoint confrontati;
- randomizzazione completa: il migliore raggiunge circa **31/50 = 62%**;
- randomizzazione parziale: circa **98% a scala 0,75**, **94% a 0,80**, **78% a 0,85**.

**[R]** La scala 0,75 viene scelta come compromesso pratico: copre molta variabilità senza perdere l'affidabilità necessaria per proseguire con il task completo.

### 5–7 settembre 2026 — phase router e nascita di Phase 2

**5 settembre — phase router basato su TCC. [G]**  
Il commit `afc1dd4` aggiunge l'esperimento di routing raw-TCC: il frame corrente viene confrontato con i prototipi iniziali delle fasi e l'encoder viene scelto in base alla distanza. `abbcfb8` imposta come default il **primo frame soltanto**. `76f797f` aggiunge randomizzazione parziale durante l'evaluation.

Questa è la prima implementazione della futura logica gerarchica:

```text
Phase 0 → successo rilevato → selezione encoder/fase → Phase 1 → …
```

L'obiettivo di lungo periodo è permettere anche biforcazioni, per esempio Phase 1 → Phase 2.1 oppure Phase 2.2 in base allo stato visivo raggiunto.

**5–6 settembre — definizione e raccolta dati di Phase 2. [G][A]**  
Il commit `039221d` implementa il workflow Phase 2: partenza da due anelli rossi e due blu e rilascio di tutti gli anelli al centro della board. Il centro operativo è collocato sopra la superficie per evitare collisioni non fisiche con il cilindro della board. Il commit `a83a019` corregge il masking deterministico di Phase 2.

Sul server risultano:

- training TCC Phase 2 terminato il 6 settembre circa alle 00:30;
- dataset BC raw e decision-only creati il 6 settembre;
- primo PPO Phase 2 da 5M step eseguito tra il 6 e il 7 settembre.

**7 settembre — ottimo risultato nella configurazione guidata/fissa. [A]**  
La comparazione `phase2_ppo_compare_fixed_30ep_seed42` dà **30/30 successi** a tutti i checkpoint e anche al BC decision-only. È un risultato reale, ma ottenuto con una rappresentazione dell'azione e un masking molto più informativi di quelli minimali sperimentati in seguito; non misura quanto la policy abbia autonomamente appreso la strategia completa.

**7 settembre — prime inferenze end-to-end Phase 0→1→2. [A][R]**
Il router raw-TCC sul primo frame raggiunge **53/60 = 88,3%**; la versione con prototipi iniziali ripuliti raggiunge **108/120 = 90%**, con accuratezza per fase 70%/100%/100%. Il primo smoke full-task fallisce perché la seconda query del router confonde Phase 2 con Phase 0. L'introduzione del *transition parking* — riportare i controller in uno stato neutro prima di acquisire il frame di routing — rende corrette entrambe le transizioni nel run successivo.

Vengono quindi completati tre smoke end-to-end deterministici: uno in configurazione fissa e due con inizializzazione Phase 0 randomizzata a scala 0,75, seed 1337 e 2025. Sono prove di integrazione riuscite, non una stima del success rate del task completo: ciascuna directory contiene un solo episodio e usa le policy guided/fixed disponibili all'epoca.

### 8–10 settembre 2026 — dalla maschera guidata alla maschera minimale

**9 settembre — prototipo di cambio fase deciso dalla distanza TCC. [A][R]**
`eval_full_task.py` viene esteso con un detector di success basato su soglia, debounce di cinque step e minimo di venti step. Un episodio completo termina con successo sia usando transizioni simboliche sia usando soglie TCC. Nel trace simbolico i tre trigger coincidono con task fisicamente completi; nel trace TCC i tre trigger avvengono prima che la condizione simbolica risulti completa (`symbolically_complete_at_trigger = 0/3`). Il test dimostra che TCC può governare il cambio di fase, ma non valida ancora le soglie come sostituto affidabile del success geometrico.

**8–9 settembre — domanda sperimentale centrale. [A][R]**  
Viene messo in discussione il masking che incorporava parti sostanziali della soluzione, ad esempio target privilegiati e assegnazioni strategiche dei bracci. Si distingue tra:

- vincoli **fisici/semantici**, necessari per impedire comandi senza significato o irraggiungibili;
- vincoli **strategici**, che rivelano alla policy dove portare gli anelli o quale braccio debba usarli.

Nasce la modalità `minimal`: mantenere solo le capacità fisiche della State Machine e lasciare che sia TCC/PPO a imparare la strategia.

**10 settembre — Phase 0 regge la riduzione del masking. [A]**  
Il fine-tuning Phase 0, randomizzazione 0,75 e maschera minimale, dura circa 3M step:

- successo iniziale della finestra: **14%**;
- successo finale: **91%**;
- picco: **100%**.

Report: `artifacts/reports/random_sb3_log_sim_phase_0_adaptive075_minimalmask_finetune_3m/`.

**10 settembre — Phase 1 collassa con la stessa ricetta. [A]**  
Il primo fine-tuning minimale Phase 1 viene fermato a circa 1,43M step:

- successo iniziale: **46%**;
- successo finale: **0%**;
- timeout finale: **100%**;
- lunghezza media finale: **600 step**.

La variante goal-agnostic da 1,5M step scende dal **49%** al **35%**. Il fatto decisivo è che la ricetta riuscita su Phase 0 non si trasferisce automaticamente a Phase 1.

### 11–16 settembre 2026 — diagnosi del collasso di Phase 1

**11 settembre — curriculum e soft mask non bastano. [A]**  
Una soft mask con penalità 8 parte circa dal 93% ma termina al 50%; un curriculum adaptive minimal termina circa al 75%. La `manipulability_scale` regola quanto della libertà d'azione minimale è stata aperta, mentre la soft penalty disincentiva, senza vietarle, le azioni sconsigliate dalla maschera guidata.

**14 settembre — dataset nativo minimale. [A]**  
Viene raccolto `dataset_supervisionato_phase1_native_minimal_160.npz`: dimostrazioni generate e verificate direttamente sotto maschera minimale, non semplicemente rietichettate. Il BC nativo raggiunge circa il **48%** nell'evaluation deterministica completa. È sufficiente come inizializzazione, ma non ancora robusto.

**14–15 settembre — PPO conservativo e orizzonte temporale. [A][R]**  
I tentativi PPO standard/conservativi collassano dopo l'unfreeze. Si osserva che un successo Phase 1 richiede mediamente oltre 400 step, mentre con `gamma=0.99` una reward terminale lontana pesa pochissimo sulle prime decisioni. Inoltre gli stati intermedi corretti possono essere visivamente più lontani dal goal rispetto a fallimenti statici.

Esperimenti:

- PPO conservativo: successo finale **0%**;
- `gamma=0.999`: circa **52% → 3%**;
- `gamma=0.999`, `gae_lambda=0.999`: circa **55% → 46%**.

L'aumento di gamma migliora l'orizzonte del credito, ma non risolve da solo il drift della policy né la scarsità delle decisioni strategiche utili.

**16 settembre — ancoraggio al BC e replay iniziale. [A]**  
Si provano:

- penalità KL verso il BC originale: circa **56% → 39%**;
- replay BC generico: circa **54% → 35%**.

La KL dei log PPO, da sola, misura la distanza dalla policy dell'aggiornamento precedente, non dal BC originale. L'ancoraggio esplicito e il replay cercano di preservare le competenze della dimostrazione, ma una distribuzione replay non mirata non è ancora sufficiente.

### 17–18 settembre 2026 — replay stratificato e recovery: svolta di Phase 1

**17 settembre — replay decisionale stratificato. [A]**  
Il replay viene concentrato sulle transizioni semanticamente informative invece di campionare in modo uniforme tutti gli step low-level. Il probe con replay al 50% completa 400k step; il miglior checkpoint raggiunge circa **52%** sul task completo di base.

Concettualmente vengono mantenuti due segnali nello stesso training:

```text
rollout on-policy PPO ──→ migliorare il ritorno TCC/terminale
batch di dimostrazioni ─→ preservare le decisioni esperte importanti
```

I due contributi producono gradienti sommati nell'ottimizzazione. Il “canale separato” non è un secondo agente né un buffer off-policy PPO: è una loss supervisionata aggiuntiva su esempi dimostrativi, concettualmente vicina a DAPG.

**17 settembre — analisi degli stati di recovery. [A]**  
Si testa esplicitamente se la policy può recuperare da distribuzioni intermedie `3–1`: tre anelli su un piolo e uno sull'altro. Entrambe le direzioni risultano fisicamente eseguibili. Viene quindi creato un dataset recovery pesato che sovracampiona queste situazioni rare.

**17–18 settembre — PPO recovery mixed. [A]**  
Un ulteriore PPO da 300k combina task completo e replay recovery. Le evaluation deterministiche finali del checkpoint selezionato danno:

- task completo `stack_green`: **49/50 = 98%**;
- ripetizione/ablation set: **48/50 = 96%**;
- recovery `red3_blue1`: circa **70%**;
- recovery `red1_blue3`: circa **55–65%**, secondo checkpoint/run.

La performance completa osservata è quindi **96–98%** con maschera minimale. I due run usano lo stesso seed e il 97/100 aggregato è descrittivo, non 100 prove indipendenti.

**18 settembre — ablation senza replay. [A]**  
La comparazione conferma che la linea con replay è quella da mantenere. Il replay non “risolve” il task al posto di PPO: impedisce che l'ottimizzazione distrugga le decisioni utili già presenti e rinforza stati rari che il rollout visita troppo poco.

### 18–24 settembre 2026 — applicazione delle lezioni a Phase 2

**18–19 settembre — primo adattamento minimale Phase 2. [A]**  
Il training adaptive minimal senza replay completa 3M step; nella successiva evaluation fully-minimal arriva circa al **15%**. Un probe minimale con replay da 400k porta il miglior checkpoint circa al **20%**. La pipeline “partire dal vecchio checkpoint guidato e togliere la maschera” è insufficiente.

**19–20 settembre — BC nativo minimale fisso. [A]**  
Viene creato un dataset Phase 2 nativo minimale da 160 episodi. La pipeline viene però fermata perché il BC non raggiunge il gate del 90%. Questo gate si rivela troppo severo per un task in cui le dimostrazioni e la distribuzione delle decisioni sono più varie.

**21 settembre — analisi della composizione del dataset. [A]**  
Si provano separatamente:

- BC decision-only;
- BC con bilanciamento decisioni + reach.

Lo scopo è evitare che i molti step ripetitivi della State Machine soffochino i pochi punti in cui la policy deve scegliere davvero target, braccio e transizione.

**22 settembre — guided BC, bimanualità e recovery. [A]**  
Una pipeline guided→minimal e un primo BC guided vengono fermati dai gate. Un dataset guided bimanuale completa invece il training, seguito da un BC che include recovery. Questi esperimenti verificano che la bimanualità è apprendibile e forniscono un teacher più ricco, ma non eliminano il divario tra guided e minimal.

**23 settembre — PPO guidato e BC asincrono bimanuale. [A]**  
Il probe PPO guidato da 400k arriva a circa **90%** nel training, ma il ranking sceglie ancora il BC, con **93,3%**: in questa configurazione PPO non migliora il teacher.

Il nuovo BC asincrono bimanuale ottiene in evaluation guidata:

- task completo: **41/50 = 82%**;
- recovery `center2`: **30/30 = 100%**;
- recovery `center3`: **29/30 = 96,7%**.

Lo stesso BC, valutato direttamente con maschera minimale completa, ottiene però solo **8/30 = 26,7%**. È la misura più chiara del distribution shift prodotto dal cambio di maschera.

**23 settembre — PPO adaptive minimal da 400k. [A]**  
Il training interno passa circa da 90% a 85%, ma la evaluation fully-minimal finale è **9/30 = 30%**: miglioramento piccolo rispetto al BC, non ancora una policy robusta.

**24 settembre — resume da 1,5M. [A]**  
Il curriculum raggiunge soltanto `manipulability_scale = 0.6` con soft penalty 8. Il successo di training passa circa da 94% a 66%. Nella evaluation con maschera completamente minimale:

- checkpoint finale: **14/30 = 46,7%**;
- BC di partenza: **8/30 = 26,7%**.

Dei 16 fallimenti del checkpoint finale, 14 terminano nello stato `center3`: manca un solo anello. La policy ha quindi appreso gran parte della sequenza, ma il completamento dell'ultimo trasferimento è ancora fragile.

### 25 settembre 2026 — nuova pipeline Phase 2 ispirata alla soluzione Phase 1

**25 settembre — raccolta nativa minimale densa e recovery stratificato. [A]**  
Si abbandona come linea principale il semplice passaggio guided→minimal e si replica più fedelmente la ricetta che ha funzionato su Phase 1:

1. dimostrazioni generate direttamente a `minimal`, `manipulability_scale = 1`;
2. combinazione di esecuzioni seriali e bimanuali;
3. verifica che ogni azione dimostrativa sia valida sotto la stessa maschera usata dal learner;
4. BC sul task completo;
5. PPO con freeze iniziale e replay supervisionato;
6. dataset recovery `center2` e soprattutto `center3` sovracampionato;
7. secondo PPO recovery-mixed e ranking deterministico.

Gli smoke test hanno verificato:

- bimanuale completo: **1164/1164** azioni mask-valid;
- seriale completo: **1429/1429**;
- recovery `center2`: **561/561**;
- recovery `center3`: **342/342**.

Sono state raccolte 112 dimostrazioni dense bimanuali e 48 seriali, poi suddivise in 128 episodi train e 32 valid. Il dataset recovery pianificato usa pesi indicativi `full ×1`, `center2 ×3`, `center3 ×8`.

**25 settembre — completamento della pipeline e limite del recovery fine-tuning. [A]**
La pipeline `phase2_native_minimal_phase1_recipe_pipeline` termina integralmente: BC, PPO full-task, PPO recovery-mixed, report e ranking deterministici. Sul primo ranking full-task da 50 episodi:

- BC nativo minimale: **35/50 = 70%**;
- checkpoint PPO finale: **38/50 = 76%**;
- checkpoint a 384k step, selezionato: **39/50 = 78%**;
- frazione di step bimanuali del checkpoint selezionato: circa **14,6%**.

Il successivo fine-tuning recovery da 300k non migliora la policy completa. In un secondo ranking il checkpoint full-task di partenza ottiene **34/50 = 68%**, mentre il migliore del recovery PPO ottiene **29/50 = 58%** e il finale **23/50 = 46%**. Sugli stati sintetici `center2` e `center3`, il checkpoint full-task raggiunge invece **28/30 = 93,3%** in entrambi i casi. La decisione è quindi non propagare il checkpoint recovery e mantenere il checkpoint full-task selezionato.

La differenza **78% vs 68%** osservata in due ranking nominalmente equivalenti segnala inoltre che una singola evaluation da 50 episodi non è sufficiente per caratterizzare stabilmente Phase 2.

### 26 settembre 2026 — evaluation multi-seed di Phase 2

**26 settembre — 150 episodi su cinque seed. [A]**
Il checkpoint `phase2_native_minimal_full_best_50ep.zip` viene valutato per 30 episodi sui seed 7, 101, 202, 303 e 404:

| Seed | Successi | Bimanual step fraction |
|---:|---:|---:|
| 7 | 25/30 = 83,3% | 18,2% |
| 101 | 27/30 = 90,0% | 20,0% |
| 202 | 27/30 = 90,0% | 18,8% |
| 303 | 26/30 = 86,7% | 17,2% |
| 404 | 22/30 = 73,3% | 13,2% |

Il totale descrittivo è **127/150 = 84,7%**. Questo è il dato più solido disponibile per Phase 2 minimale, pur essendo ancora l'evaluation di un singolo checkpoint di training e non una stima su seed di addestramento indipendenti.

### 27 settembre 2026 — confronto visuale, bimanualità e protocollo controllato

**27 settembre — inferenze video comparabili per tutte le fasi. [A][R]**
Vengono prodotti tre video deterministici per ciascuna fase sotto maschera minimale:

- Phase 0: **3/3 successi**, 196–205 step, **550/603 = 91,2%** di step bimanuali;
- Phase 1: **3/3 successi**, 343–376 step, **0/1063** step bimanuali;
- Phase 2: **2/3 successi**, successi in 294 e 296 step, **263/1190 = 22,1%** di step bimanuali complessivi.

L'ispezione del fallimento Phase 2 mostra tre anelli già completati e il destro che riafferra un anello completato, poi rimane impegnato in una `reach board_center` che termina in recovery/timeout. La maschera minimale impedisce azioni fisicamente invalide, ma permette correttamente questa scelta strategicamente sfavorevole. Emerge anche un distribution shift: gli stati recovery sintetici `center3` partono con bracci liberi e State Machine pulita, mentre il fallimento reale raggiunge `center3` con un anello trattenuto e una primitive già bloccata.

**27 settembre — audit delle dimostrazioni e reinterpretazione della bimanualità. [A][R]**
L'analisi dei dataset BC chiarisce che:

- Phase 0 contiene **0%** azioni bimanuali ma usa i bracci in modo bilanciato, circa 49/51%;
- Phase 1 contiene **0%** azioni bimanuali, ma è sbilanciata circa 75/25 verso il sinistro;
- il replay recovery Phase 1 rimane **0%** bimanuale e aumenta lo sbilanciamento a circa 82/18;
- la più recente Phase 2 non è un caso puro di emergenza, perché il suo dataset combina dimostrazioni asincrone bimanuali e seriali.

La spiegazione proposta per Phase 1 non è quindi semplicemente “mancano esempi bimanuali”. Phase 0 dimostra che PPO può sviluppare bimanualità emergente. In Phase 1, però, tutti gli anelli partono dalla stessa pila: inizialmente soltanto il top ring è manipolabile e il parallelismo utile richiede un pipelining asincrono dopo il primo grasp. Dataset sbilanciato, replay seriale, budget PPO inferiore e soluzione seriale già robusta rendono questa strategia difficile da scoprire.

**27 settembre — avvio del protocollo scientificamente controllato. [A][R]**
Si riconosce che le policy migliori correnti non possono essere confrontate causalmente: Phase 0 ha ricevuto milioni di step senza replay finale, Phase 1 usa replay/recovery conservativo e Phase 2 include dimostrazioni bimanuali. Vengono quindi separati due obiettivi:

1. **best-performing system**, nel quale sono ammessi curriculum, replay e ricette specifiche per fase;
2. **controlled baseline**, nel quale pipeline e budget sono identici e l'unica differenza intenzionale è il task.

Il protocollo `controlled_serial_minimal_1m` viene avviato sul GPU 0 con questa configurazione preregistrata:

- 160 nuove dimostrazioni riuscite, seriali e native-minimal per fase;
- ordine casuale e braccio casuale tra quelli fisicamente raggiungibili;
- split identico 128 train / 32 validation;
- BC identico: 20 epoch, batch 256, learning rate `1e-3`, seed 42;
- PPO senza replay, KL o curriculum, per non ancorare la policy all'esperto seriale;
- 1M transizioni per fase e seed di training 101, 202 e 303;
- `gamma = 0.999`, `gae_lambda = 0.999`, freeze di 150k e schedule LR `3e-6 → 3e-5 → 3e-6`;
- evaluation del checkpoint finale, senza selezione post-hoc del best, su 100 episodi deterministici per run.

Prima dell'avvio viene corretto un problema di riproducibilità: `parallel_run.py` aveva il seed fissato internamente a 42, per cui repliche nominalmente diverse non sarebbero state indipendenti. La strumentazione viene estesa con invalid command episodici, step bimanuali, attività sinistra/destra, cambi di joint action e slot con braccio libero. Lo smoke test produce 702 transizioni Phase 0, tutte mask-valid e nessuna azione esperta bimanuale; i test automatici passano **58/58**. Al momento dell'aggiornamento la pipeline è in `COLLECTING_BC` per Phase 0.

### 1–6 ottobre 2026 — Phase 3: dal teacher seriale alla bimanualità robusta

**1 ottobre — definizione operativa della Phase 3. [A][R]**
La fase parte con tutti e quattro gli anelli al centro della board (`center4`) e termina quando ciascuno dei quattro pioli colorati contiene esattamente un anello. Colore dell'anello e ordine dei trasferimenti sono irrilevanti. La maschera rimane minimale: conserva i soli vincoli fisici e semantici, incluso il fatto che il braccio destro non raggiunge il piolo rosso, senza imporre una strategia di assegnazione.

**1–2 ottobre — dataset TCC e compatibilità GPU. [A]**
Vengono raccolti 160 video train e 20 valid per Phase 3, per un totale di 12.123 e 1.495 frame PNG. Il primo pretraining TCC fallisce sulla RTX 5090 perché la versione PyTorch dell'ambiente XIRL non supporta l'architettura `sm_120`. Dopo aver verificato che il processo presente sulla RTX 4090 apparteneva al progetto separato di mobile robotics, quel servizio viene arrestato e TCC viene eseguito sulla 4090 fino a 4.000 iterazioni.

L'audit dei checkpoint TCC seleziona `2600.ckpt`, corrispondente al miglior Kendall tau di validazione osservato (circa **0,574**), invece dell'ultimo `4001.ckpt`. Viene corretto un disallineamento iniziale per assicurare che collector, PPO ed evaluation usino tutti lo stesso encoder.

**2–3 ottobre — i primi BC mostrano due limiti opposti. [A][R]**
Un dataset con ordine e assegnazioni casuali raggiunge circa 98,5% di accuratezza supervisionata, ma **0/20** in evaluation deterministica: label multimodali e mismatch operativo rendono l'accuratezza BC poco predittiva del task. Un teacher seriale canonico, solo braccio sinistro e ordine fisso, è invece coerente ma limitato:

- BC 20 epoch: **3/20 = 15%**;
- BC 80 epoch: **13/20 = 65%** nel gate iniziale;
- dopo PPO con replay, miglior checkpoint seriale: **38/50 = 76%**;
- frazione bimanuale: **0%**.

**3–4 ottobre — scheduler esperto bimanuale asincrono. [A][R]**
Poiché gli anelli partono impilati al centro, due reach simultanee iniziali non sono fisicamente utili. Il nuovo scheduler usa quindi un pipeline asincrono: il sinistro afferra il primo anello ed espone il successivo; mentre lo trasferisce verso un piolo, il destro può raggiungere il nuovo top ring. La sequenza viene ripetuta per due coppie.

Lo smoke test ottiene **4/4 successi**, 278–302 step, circa **30,2%** di righe bimanuali e **1153/1153** azioni valide sotto la maschera minimale. Viene poi raccolto un dataset puro bimanuale da 160 episodi e 45.059 transizioni. Il BC a 80 epoch raggiunge **19/20 = 95%**, con circa **28,1%** di attività bimanuale aggregata e 278,8 step medi nei successi.

**4–6 ottobre — PPO ancorato e ranking deterministico finale. [A]**
Il BC bimanuale viene sottoposto a un probe PPO da 400k con replay supervisionato. Nel training il successo passa dal 100% della prima finestra al 95% dell'ultima, con 5% di timeout e circa 301 step medi. Il ranking conclusivo su 50 episodi deterministici produce:

| Checkpoint | Successo | Step bimanuali | Step medi nei successi |
|---|---:|---:|---:|
| PPO finale | **48/50 = 96%** | **28,1%** | 282,0 |
| BC bimanuale 80 epoch | **48/50 = 96%** | **28,1%** | 282,0 |
| PPO best terminal distance | 47/50 = 94% | 27,8% | 282,2 |
| PPO 384k | 47/50 = 94% | 27,8% | 282,2 |

Il checkpoint distribuito viene quindi copiato dal PPO finale. Scientificamente, però, questa prova **non dimostra un miglioramento prodotto da PPO rispetto al BC**, perché entrambi ottengono 96% e mostrano sostanzialmente lo stesso comportamento. Dimostra invece che il nuovo teacher/dataset bimanuale porta la Phase 3 dal precedente 76% seriale al 96%, introduce bimanualità reale e che il fine-tuning PPO non distrugge la competenza acquisita.

### 6–9 ottobre 2026 — Phase 4 e composizione del task completo 0→4

**6 ottobre — definizione e pipeline canonica della Phase 4. [A][R]**
La Phase 4 parte dalla condizione di successo di Phase 3: un anello qualsiasi su ciascuno dei quattro pioli colorati. Il goal è rimuovere tutti gli anelli dalla board, depositandone esattamente due nella regione esterna destra e due nella regione esterna sinistra. Le due regioni vengono rappresentate riutilizzando due token target non necessari al task (`peg_gray` e `peg_gray1`), senza aumentare la dimensionalità dell'azione o dell'osservazione.

Vengono introdotti i target fisici `outside_right` e `outside_left`, con due slot per regione per evitare sovrapposizioni. Dataset TCC, teacher bimanuale asincrono, BC e PPO seguono la stessa pipeline recente di Phase 3. Il checkpoint TCC selezionato è `2600.ckpt`.

Il ranking canonico su 50 episodi deterministici produce:

| Checkpoint Phase 4 | Successo | Step bimanuali | Step medi nei successi |
|---|---:|---:|---:|
| PPO best terminal distance | **49/50 = 98%** | **88,8%** | 184,6 |
| PPO finale | **49/50 = 98%** | **88,8%** | 184,6 |
| PPO 384k | **49/50 = 98%** | **88,8%** | 184,6 |
| BC bimanuale 80 epoch | 36/50 = 72% | 93,2% | 195,3 |

Questa è un'evidenza forte per la Phase 4 **standalone dalla posa canonica**, ma non ancora per l'interfaccia con la Phase 3.

**7 ottobre — estensione del router e prima inferenza Phase 0→4. [A]**
Il router TCC viene esteso agli encoder Phase 3 e Phase 4 mantenendo l'ipotesi del solo primo frame come start prototype. Nell'episodio controllato a stato iniziale fisso, le quattro transizioni sono tutte selezionate correttamente (`4/4`): Phase 0→1, 1→2, 2→3 e 3→4. Le prime quattro fasi vengono completate, ma la policy Phase 4 entra in un loop e il full task termina senza successo.

Nel test con randomizzazione iniziale degli anelli di Phase 0 a scala `0.75`, l'episodio si arresta invece già in Phase 2. Questo separa due problemi distinti:

- robustezza delle singole policy sotto la distribuzione randomizzata;
- distribution shift articolare fra la fine di una fase e l'inizio della successiva.

**7–8 ottobre — adattamento parked e test di homing. [A][R]**
Viene raccolto un dataset Phase 4 dalla posa articolare prodotta dal parking inter-fase. Un adattamento BC bilanciato migliora l'accuratezza offline parked, ma ottiene **0/5** in closed loop e viene fermato prima di PPO. Il comportamento osservato sposta correttamente il primo anello e poi ripete comandi verso la regione esterna senza aver afferrato il successivo.

Si implementa quindi un comando fisico `home` nella State Machine e l'opzione `--home_before_phase phase_4` nel full-task evaluator. Nel test fisso:

- Phase 0 completa in 662 step;
- Phase 1 in 351;
- Phase 2 in 378;
- Phase 3 in 285;
- router corretto in tutte le 4 transizioni;
- homing completato in 124 step low-level;
- Phase 4 fallisce dopo altri 547 step.

Il semplice ritorno cartesiano degli end-effector non ripristina necessariamente la stessa configurazione articolare/orientamento vista nel training canonico.

**8 ottobre — correzione del reset articolare e curriculum canonica→parked. [A][R]**
L'analisi del vecchio `--phase_entry_park_reset` individua un errore concettuale importante: modificare `ArticulationCfg.init_state.joint_pos` cambia anche `default_joint_pos`. Poiché l'osservazione usa `joint_pos_rel`, la posa parked veniva nuovamente osservata come zero e non riproduceva il full task. Il reset viene corretto scrivendo fisicamente le articolazioni nella posa interpolata, mantenendo la posa canonica come riferimento osservativo.

Viene avviato un curriculum success-gated da 800k step:

- scale previste: `0.00 → 0.25 → 0.50 → 0.75 → 1.00`;
- una posa viene campionata uniformemente fra canonica e difficoltà corrente;
- avanzamento dopo 128 episodi con almeno 85% di successo;
- replay BC canonico con coefficiente `0.1`;
- learning rate `2e-6`, maschera minimal, `gamma=GAE=0.999`.

Il curriculum supera `0.00` a 38.400 step ma resta bloccato a `0.25`. Su 1.933 episodi, il successo finale di training è 52%. La decomposizione per interpolazione mostra una frontiera netta:

| Interpolazione canonica→parked | Successo durante il curriculum |
|---|---:|
| `0.00–0.05` | 67,0% |
| `0.05–0.10` | 61,2% |
| `0.10–0.15` | 41,0% |
| `0.15–0.20` | 43,4% |
| `0.20–0.25` | **7,9%** |

L'evaluation finale conferma che il curriculum preserva la competenza canonica ma non apprende l'interfaccia parked:

| Distribuzione di evaluation | Originale canonico | PPO best curriculum | PPO finale curriculum |
|---|---:|---:|---:|
| canonica | 49/50 = 98% | 49/50 = 98% | 49/50 = 98% |
| parked esatta | 1/50 = 2% | 0/50 = 0% | 0/50 = 0% |

L'esperimento è quindi negativo, ma localizza quantitativamente il distribution shift: il salto `0→0,25` è troppo ampio e PPO rimane quasi ancorato alla policy iniziale (`KL` circa `10⁻⁶`). La prosecuzione proposta usa incrementi molto più piccoli, per esempio `0,05`, e una misura del successo concentrata sulla frontiera corrente.

**9 ottobre — consolidamento di report, video e comandi riproducibili. [A]**
Viene creato `RISULTATI_E_COMANDI_INFERENZA.md`, che raccoglie i path verificati dei grafici principali, le evaluation deterministiche, i video positivi e diagnostici, e i comandi foreground per inferenza full-task e Phase 0–4. Phase 0 viene sempre lanciata con randomizzazione `0.75`; Phase 1–4 conservano le rispettive condizioni iniziali semantiche.

**13–14 ottobre — BC parked da zero, full-task e handoff realmente asincrono. [A][R]**
Un nuovo BC Phase 4 viene addestrato da inizializzazione casuale su 80 dimostrazioni raccolte dalla posa parked esatta e dal suo piccolo intorno articolare. Il successivo probe PPO da 400k con replay non modifica sostanzialmente il comportamento del BC. Nei ranking deterministici il checkpoint selezionato ottiene **49/50 = 98%** dalla posa parked esatta e **50/50 = 100%** nel piccolo neighborhood. BC e checkpoint PPO mostrano però metriche quasi identiche: circa 180 step per successo e oltre il 91% di step bimanuali. Il risultato dimostra robustezza nella distribuzione sintetica valutata, non ancora nel vero handoff.

Il test completo con randomizzazione iniziale Phase 0 a scala `0.75` completa Phase 0, 1, 2 e 3 rispettivamente in 388, 352, 495 e 314 step; il router seleziona correttamente tutte e quattro le transizioni. La Phase 4 entra invece in timeout dopo 700 step. L'ispezione del log e del video localizza la prima divergenza: nella valutazione isolata entrambi i bracci eseguono `grasp` intorno allo step 30, mentre nel full task il destro ha già afferrato l'anello e il sinistro è ancora in `reach`. La policy, addestrata su una coreografia quasi sincrona, porta l'anello destro verso `outside_right` ma non emette il `release` mentre l'altro braccio è indietro; successivamente entra in un ciclo di reach/idle. La maschera minimal considera queste azioni fisicamente valide e quindi non può correggere la scelta strategica.

Viene quindi introdotto `scripts/collect_phase4_true_handoff_demos.py`. A differenza dei reset articolari sintetici, il collector esegue la policy PPO Phase 3 selezionata, conserva esclusivamente i successi simbolici, applica lo stesso parking cartesiano di `eval_full_task.py` e registra un teacher Phase 4 indipendente per braccio. Lo smoke test raccoglie **4/4** dimostrazioni Phase 4 riuscite, 178 step ciascuna, per **712/712 azioni valide** sotto maschera minimal. Ogni traiettoria contiene 13 decisioni congiunte distinte, invece delle circa 8 della coreografia perfettamente sincronizzata: questa è evidenza diretta che il nuovo dataset copre gli stati asincroni mancanti. Parte quindi la raccolta principale da 80 handoff riusciti (`phase4_true_phase3_handoff80_seed20261015.npz`); i rollout in cui Phase 3 fallisce vengono scartati e non trasformati artificialmente in ingressi validi Phase 4.

**15 ottobre — i primi adattamenti al true handoff non risolvono il problema. [A][R]**
Il BC adattato e i successivi probe PPO/replay migliorano la copertura offline ma rimangono fragili nel passaggio reale Phase 3→4. Le principali evaluation deterministiche ottengono **0/10** per il vecchio BC, **0/10** per il PPO da 400k e **2/10** per i migliori checkpoint full-state/empirical-bank. Rendere stocastica soltanto Phase 4 porta a **7/20 = 35%**, ma non produce la policy deterministica robusta desiderata. Questo risultato sposta l'attenzione dall'ottimizzazione PPO alla provenienza delle dimostrazioni.

L'ispezione del dataset BC originale di Phase 3 mostra che conserva osservazioni, azioni e indici decisionali, ma non le pose articolari e gli stati fisici completi degli anelli. I terminali non possono quindi essere ricostruiti fedelmente copiando righe dal file `.npz`. La soluzione scelta è rieseguire il **teacher bimanuale originale di Phase 3**, catturare i suoi terminali fisici dopo il parking e, da ciascuno di questi, etichettare Phase 4 con il teacher minimal.

**16 ottobre — dataset teacher-terminal e nuovo BC Phase 4. [A]**
`collect_phase4_true_handoff_demos.py` viene esteso con `--phase3_source teacher` e con uno scheduler che riproduce il teacher bimanuale usato per il BC Phase 3. Lo smoke test completa **8/8 Phase 3** e **8/8 Phase 4**, produce 1.433 transizioni tutte compatibili con la maschera minimal e mostra già sei configurazioni anello→piolo distinte.

La raccolta principale salva:

- **160** stati terminali fisici Phase 3 dopo il parking;
- **160/160** dimostrazioni Phase 4 riuscite;
- **28.793** transizioni dense;
- tutte le **24/24 permutazioni** possibili fra i quattro anelli e i quattro pioli;
- episodi Phase 4 lunghi 178–211 step, mediana 179.

I file risultanti sono `phase4_phase3_teacher_terminal160_seed20261026_bank.npz` e `phase4_phase3_teacher_terminal160_seed20261026_demos.npz`. Il secondo viene diviso per episodio in **128 train / 32 validation**. Un nuovo BC Phase 4 viene addestrato da inizializzazione casuale per 80 epoche, batch 256 e learning rate `1e-3`, senza ereditare pesi dai precedenti tentativi. Ottiene **100% exact-action accuracy** sul train e **99,98%** sulla validation separata.

La prova decisiva usa il PPO Phase 3 realmente selezionato, parking, router TCC e il nuovo BC Phase 4 con azioni deterministiche e maschera minimal. Su 20 episodi:

- Phase 3 completa **20/20**;
- il router seleziona correttamente Phase 4 **20/20**;
- Phase 4 completa **20/20**;
- il task Phase 3→4 ottiene quindi **20/20 = 100%**;
- Phase 4 richiede in mediana 181 step, media 182,75 e massimo 215.

Il salto rispetto ai precedenti 0–35% indica che il principale collo di bottiglia era il **support mismatch del dataset BC**, non una mancanza intrinseca di capacità della policy né la necessità immediata di PPO. Il criterio di completamento di questa evaluation è però simbolico: TCC viene usato per il routing, non per dichiarare il successo della fase.

Il successivo protocollo di robustezza Phase 3→4 si conclude con **100/100 successi** su cinque seed da 20 episodi e **100/100 query del router corrette**. PPO Phase 4 resta intenzionalmente sospeso: il BC teacher-terminal è già robusto nella distribuzione di handoff misurata e non esiste ancora evidenza che un ulteriore fine-tuning RL lo migliori.

La full-task evaluation associata, eseguita su 20 episodi con randomizzazione iniziale `0.75`, successo TCC e il vecchio checkpoint Phase 2 native-minimal, ottiene invece **5/20 = 25%**. Gli altri quindici episodi terminano con otto timeout Phase 2, quattro apparenti errori di routing Phase 2→Phase 0, un timeout Phase 0, un timeout Phase 1 e una terminazione ambientale. L'analisi successiva mostra che i quattro errori del router sono in realtà trigger TCC prematuri: Phase 2 viene dichiarata conclusa nello stato a tre anelli, non ancora simbolicamente completo.

**19 ottobre — Phase 2 teacher-terminal: il BC diventa robusto al vero handoff Phase 1→2. [A]**
Per eliminare il support mismatch fra il reset standalone di Phase 2 e lo stato realmente prodotto da Phase 1, vengono raccolti **160 veri terminali Phase 1**. Il collector enumera le 24 permutazioni bottom-to-top degli anelli e le sei sequenze valide di destinazione 2-rosso/2-blu, esegue il parking inter-fase e registra poi un teacher Phase 2 asincrono. Il dataset contiene 46.771 transizioni, 24 layout fisici terminali distinti e nessuna etichetta invalida sotto maschera minimal; viene diviso per episodio in 128 train e 32 validation.

Il BC da zero `phase2_bc_phase1_teacher_terminal128_80ep_seed42.zip` raggiunge 100% di accuratezza exact-action sul train e 99,98% sulla validation. Nella valutazione deterministica matched Phase 1→2 ottiene **49/50 = 98%**. L'unico fallimento osservato è una recovery rara: entrambi gli anelli superiori vengono afferrati, una reach verso il centro non conclude, un braccio conserva l'anello blu e la policy entra in una sequenza idle/reach senza terminare.

**19–20 ottobre — il probe PPO conservativo preserva il BC ma non impara la recovery. [A][R]**
Si esegue un probe PPO da 400k con posa parked, maschera minimal, freeze dell'attore per 150k, replay BC `0.1`, anchor `KL(BC||PPO)=0.01` e `gamma=GAE=0.999`. Nel ranking standalone sia BC sia checkpoint PPO ottengono **49/50 = 98%** con comportamento sostanzialmente identico. Nel vero handoff il PPO selezionato ottiene 46 successi Phase 2 su 47 ingressi Phase 1 riusciti (**97,9% condizionale**), quindi non supera il BC; tre ulteriori episodi erano già falliti in Phase 1.

I log chiariscono perché il probe non è una dimostrazione di miglioramento RL: replay exact-accuracy resta 1, la policy rimane quasi uguale al BC e la KL numerica diventa `inf`, contaminando KL loss e loss totale. La causa è l'underflow a probabilità zero nella distribuzione categorica mascherata. La KL viene quindi riscritta in log-space float64 con rapporto logaritmico limitato e test espliciti di finitezza; l'intera suite raggiunge **75/75 test**.

**20 ottobre — avvio del curriculum RL da veri stati di recovery Phase 2. [A][R]**
Per non sostituire RL con altre etichette esperte, viene introdotta una banca che salva esclusivamente **stati fisici**, non azioni: pose e velocità articolari, pose degli anelli, supporti simbolici, livelli dello stack e anello eventualmente ancora afferrato. La policy BC viene campionata stocasticamente per visitare un intorno del comportamento nominale; gli stati vengono catturati durante timeout reali oppure in traiettorie stagnanti oltre 350 step con un anello ancora in pinza. Il reset empirico, nato per Phase 4, viene generalizzato a Phase 2 e ricostruisce coerentemente anche attachment e stato delle pinze.

La nuova pipeline `run_phase2_recovery_curriculum_pipeline.sh` usa:

- 50% reset parked nominali e 50% reset dalla banca di recovery;
- PPO per 500k step, freeze ridotto a 50k;
- replay BC debole `0.03` e KL debole `0.001`;
- learning rate `5e-7 → 2e-6 → 2e-7`;
- ranking deterministico su tutti i 64 stati di recovery;
- test di regressione su 50 episodi parked nominali.

Negli stati di recovery non vengono fornite azioni corrette: il miglioramento deve provenire dalla reward TCC e dal bonus terminale. Replay e KL hanno soltanto funzione anti-forgetting. Lo smoke test sulla RTX 5090 osserva il primo stall stocastico allo step 350 con `ring_green` ancora afferrato dal braccio destro e salva correttamente quattro stati consecutivi. Un processo di evaluation separato ricarica la banca, ripristina la pinza destra chiusa e l'attachment e riparte dalla posa fisica registrata. Il BC non risolve immediatamente lo stato e continua a ciclare dopo un primo rilascio: è quindi un reset difficile ma valido, non una duplicazione degli stati nominali. La RTX 4090 occupata da Qwen non viene modificata.

**21 ottobre — il curriculum recovery preserva il nominale ma non risolve gli stati difficili. [A][R]**
Il PPO termina 500k step, ma il ranking iniziale va in timeout perché 64 episodi sequenziali superano il limite fisso di un'ora. `rank_checkpoints.py` viene corretto per eliminare anche i discendenti Isaac rimasti vivi dopo `SIGTERM`, e il timeout per checkpoint viene portato a tre ore. Il rerun completa tutti i confronti.

Sui 64 stati fisici della banca recovery i risultati sono negativi:

- BC originale: **2/64 = 3,12%**, media 2,50 anelli al centro;
- PPO best-terminal-distance: **1/64 = 1,56%**, media 2,48 anelli;
- PPO a 384k: **0/64**, ma media 2,81 anelli e 53/64 episodi terminati con tre anelli;
- PPO finale 500k: **0/64**, media 2,63 anelli.

Il PPO a 384k acquisisce quindi progresso parziale senza apprendere il completamento dell'ultimo trasferimento. Nella valutazione nominale diretta da 50 episodi, invece, BC, PPO best, PPO 384k e PPO finale ottengono tutti **50/50**, con 290,92 step medi e comportamento deterministico sostanzialmente identico. Un precedente run matched aveva dato 49/50, perciò la differenza 98–100% va letta come variabilità della simulazione/evaluation, non come un miglioramento misurabile del PPO. Replay e KL impediscono il forgetting, ma il curriculum recovery non raggiunge il proprio obiettivo.

**25 ottobre — full task Phase 0→4 al 95% con la nuova Phase 2 teacher-terminal. [A][R]**
Si ripete il protocollo full-task del baseline da 20 episodi mantenendo seed `20261033`, randomizzazione iniziale `0.75`, azioni deterministiche, maschera minimal, router first-frame e successo TCC. L'unica sostituzione intenzionale è la policy Phase 2: il vecchio checkpoint native-minimal viene rimpiazzato dal PPO teacher-terminal conservativo da 400k, già verificato nel vero handoff Phase 1→2.

Il risultato passa da **5/20 = 25%** a **19/20 = 95%**:

- Phase 0: 20/20 completamenti;
- Phase 1: 19/20, con l'unico fallimento dovuto a timeout;
- Phase 2: **19/19**, media 303,6 step e nessun timeout;
- Phase 3: 19/19;
- Phase 4: 19/19;
- router: **77/77** selezioni corrette;
- success detector TCC: **96/96** trigger avvenuti in stati anche simbolicamente completi.

La nuova Phase 2 non si ferma più in uno stato quiescente a tre anelli: completa il quarto trasferimento in 299–310 step. Di conseguenza scompaiono sia gli otto timeout Phase 2 sia i quattro falsi errori del router osservati nel baseline. Il risultato valida la composizione operativa PPO–TCC–router end-to-end, ma non attribuisce causalmente il salto a PPO: standalone il PPO teacher-terminal e il relativo BC producono quasi lo stesso comportamento. Per separare l'effetto del dataset BC dal fine-tuning RL serve ancora una full-task ablation con il BC non fine-tuned e una valutazione multi-seed del sistema completo.

---

## 3. Evoluzione delle principali decisioni progettuali

### Reward

1. TCC/XIRL viene adottato come reward visiva densa basata sulla distanza dal goal embedding.
2. Il goal rimane la media degli embedding finali delle dimostrazioni.
3. Il 3 settembre viene ripristinato il bonus logico di successo, pari storicamente a `+100`.
4. L'analisi di Phase 1 mostra che reward densa, bonus terminale e discount devono essere letti insieme: un percorso corretto ma lungo può accumulare più costo di un fallimento statico.
5. `gamma`/GAE più lunghi aiutano il credit assignment, ma non impediscono da soli il policy drift.

### Action masking

1. Le prime pipeline usano una maschera **guided**, molto informativa.
2. Vengono aggiunti controlli di raggiungibilità, timeout e recovery per eliminare comandi fisicamente impossibili.
3. Si riconosce che assegnazioni obbligatorie di braccio, target privilegiati e divieti strategici incorporano parte della soluzione.
4. Nasce la maschera **minimal**, che mantiene soltanto vincoli fisici/semantici.
5. I risultati guided e minimal non vengono più confrontati come se misurassero la stessa capacità.

### Frequenza decisionale

All'inizio PPO emette un'azione a ogni step low-level, anche mentre una `reach` è in corso. Questo consente teoricamente di cambiare idea, ma produce moltissimi esempi ripetitivi rispetto alle poche decisioni strategiche reali. Per stabilizzare gli esperimenti recenti si privilegiano decision points e macro-azioni persistenti; la possibilità di cambiare idea durante una primitive rimane un'estensione futura, non un obiettivo eliminato.

### Behavioural Cloning e replay

1. Il BC iniziale insegna alla policy un comportamento fattibile.
2. Un dataset raccolto sotto una maschera diversa da quella di deployment crea distribution shift.
3. Il dataset viene quindi raccolto e validato nativamente sotto `minimal`.
4. Il replay supervisionato affianca PPO durante il fine-tuning per preservare le decisioni esperte.
5. Il replay uniforme è meno efficace di quello stratificato sui decision points.
6. I dataset recovery sovracampionano stati rari ma critici (`3–1` in Phase 1, `center2/center3` in Phase 2).

### Comparabilità scientifica

1. Le pipeline specifiche per fase rimangono appropriate per costruire il sistema più robusto possibile.
2. Non sono però sufficienti per attribuire una differenza di comportamento alla struttura del task: cambiano simultaneamente dati, budget, replay, curriculum e seed.
3. Il protocollo controllato del 27 settembre usa quindi BC seriali/minimal, iperparametri uguali, tre seed PPO e budget fisso di interazioni.
4. Lo stesso numero di `total_timesteps` è la variabile di budget primaria, ma vengono registrati anche episodi, cambi di azione, attività dei bracci e slot decisionali, perché le tre fasi hanno durate e affordance differenti.
5. I risultati del protocollo controllato e quelli delle policy migliori devono essere riportati in tabelle separate.

### Phase routing

1. Inizialmente le fasi vengono eseguite e valutate separatamente.
2. Viene introdotto un router TCC che confronta il frame corrente con gli start prototype.
3. Per rispettare l'ipotesi sperimentale originale, lo start prototype usa il solo primo frame.
4. Il success della fase può essere rilevato geometricamente oppure tramite soglia sulla distanza TCC.
5. Il 7 ottobre la struttura viene estesa alla sequenza Phase 0 → 1 → 2 → 3 → 4; nel probe fisso il router seleziona correttamente tutte le quattro transizioni.
6. Parking e homing rendono esplicito che selezionare correttamente la fase non basta: la policy successiva deve essere robusta anche alla distribuzione articolare prodotta dall'handoff.
7. Il nuovo BC teacher-terminal completa 20/20 passaggi deterministici Phase 3→4, ma usa successo simbolico; la robustezza multi-seed e la sequenza Phase 0→4 con successo TCC sono in valutazione.
8. La struttura resta compatibile con future biforcazioni, ma una singola evaluation positiva non basta ancora a dichiarare robusta l'intera sequenza end-to-end.

### Reset e distribuzione di ingresso delle fasi

1. Le prime evaluation standalone usano pose articolari canoniche a ogni reset.
2. Nel full task non avviene un reset fra fasi: la policy successiva eredita la configurazione prodotta dalla precedente e dal parking.
3. Il primo `phase_entry_park_reset` cambiava anche il riferimento di `joint_pos_rel`, nascondendo parte del distribution shift.
4. Dall'8 ottobre le pose parked/interpolate vengono scritte fisicamente mantenendo canonica `default_joint_pos`.
5. Phase 4 inizialmente rimane il caso aperto: 98% canonico, 0–2% dalla posa parked esatta e 0–35% nei primi veri handoff.
6. Il dataset teacher-terminal del 16 ottobre copre l'interfaccia fisica effettiva: il nuovo BC da zero raggiunge 20/20 sul passaggio deterministico Phase 3→4.
7. Questo risultato identifica il dataset di ingresso come causa principale, ma deve ancora essere confermato su 100 episodi multi-seed e nel full task TCC.

---

## 4. Stato tecnico delle fasi al 25 ottobre 2026

| Fase | Miglior evidenza comparabile | Stato |
|---|---:|---|
| Phase 0, fixed | 20/20 = 100% | robusta nella configurazione fissa |
| Phase 0, randomizzazione 0,75 | circa 98% nella selezione precedente; training minimal finale 91% | compromesso operativo scelto |
| Phase 0, randomizzazione completa | 31/50 = 62% migliore osservato | non abbastanza robusta per essere default |
| Phase 1, completa, minimal | 49/50 e 48/50 = 96–98% | migliore fase completa corrente |
| Phase 1, recovery 3–1 | circa 55–70% | inferiore al task nominale, ancora migliorabile |
| Phase 2, fixed/guided storico | 30/30 = 100% | forte ma troppo assistita per misurare autonomia strategica |
| Phase 2, fully minimal prima del nuovo dataset | 14/30 = 46,7% | miglioramento reale, ultimo anello ancora fragile |
| Phase 2, pipeline native-minimal precedente | 127/150 = 84,7% su cinque seed di evaluation | superata come checkpoint operativo dal BC teacher-terminal |
| Phase 2, BC teacher-terminal, vero handoff | 49/50 = 98% | migliore evidenza deterministica corrente; un raro stall con anello afferrato |
| Phase 2, PPO teacher-terminal 400k | 49/50 parked; 46/47 condizionale al successo Phase 1 | preserva il BC ma non migliora la recovery; KL numerica `inf` corretta dopo il run |
| Phase 2, curriculum RL da stati di recovery | BC 2/64; PPO best 1/64; PPO 384k e finale 0/64 | nominale preservato, recovery non appresa; 384k arriva spesso a tre anelli |
| Phase 3, teacher seriale minimal + PPO | 38/50 = 76% | baseline coerente ma monomanuale |
| Phase 3, teacher bimanuale minimal, BC | 48/50 = 96% | forte baseline deterministica; 28,1% di step bimanuali |
| Phase 3, teacher bimanuale minimal, PPO finale | 48/50 = 96% | checkpoint distribuito; preserva ma non supera il BC |
| Phase 4, BC bimanuale canonico | 36/50 = 72% | bimanuale ma meno robusto del PPO |
| Phase 4, PPO canonico minimal | 49/50 = 98% | molto robusta standalone; 88,8% di step bimanuali |
| Phase 4, posa parked esatta | originale 1/50; curriculum 0/50 | distribution shift articolare ancora irrisolto |
| Curriculum Phase 4 canonica→parked | fermo a scala 0,25; 52% finale | salto iniziale troppo ampio, competenza canonica preservata |
| Phase 4, primi veri handoff | 0/10 BC, 0/10 PPO, 2/10 best deterministico, 7/20 stocastico | adattamento/replay insufficienti |
| Phase 4, BC teacher-terminal | validation exact-action 99,98%; vero handoff 20/20 | migliore evidenza corrente; PPO non ancora giustificato |
| Router Phase 0→4, probe fisso | 4/4 transizioni corrette | routing riuscito, execution end-to-end fallisce in Phase 4 |
| Full task randomizzazione Phase 0 = 0,75 | baseline 5/20; nuova Phase 2 teacher-terminal PPO 19/20 | migliore evidenza end-to-end; da confermare multi-seed |
| Robustezza Phase 3→4 multi-seed | 100/100 e router 100/100 | robusta nella distribuzione teacher-terminal misurata |
| Protocollo controllato Phase 0/1/2 | completato per 3 seed da 1M | curve aggregate di successo e bimanualità disponibili |

Le percentuali non vanno mescolate senza riportare almeno: seed, stato iniziale, modalità di maschera, randomizzazione, checkpoint e numero di episodi.

---

## 5. Artifact principali da conservare

### Phase 0

- Evaluation fixed: `artifacts/evaluations/phase0_fixed_reachguard_20ep/`
- Ranking random completo: `artifacts/evaluations/phase0_adaptive_random_rank_50ep_retry/`
- Report minimal/random 0,75: `artifacts/reports/random_sb3_log_sim_phase_0_adaptive075_minimalmask_finetune_3m/`

### Phase 1

- BC nativo minimal: `artifacts/checkpoints/bc/phase1_bc_native_minimal_160.zip`
- Checkpoint finale selezionato: `artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_1_native_minimal_recovery_mixed_replay_probe_300k.zip`
- Evaluation primaria: `artifacts/evaluations/phase1_recovery_probe_stack_green_50ep/`
- Evaluation ripetuta/ablation: `artifacts/evaluations/phase1_noreplay_ablation_stack_green_50ep/`
- Report completo: `artifacts/reports/phase1_complete_deterministic_evaluation/`
- Video deterministici: `artifacts/evaluations/phase1_complete_deterministic_videos/`

### Phase 2

- Primo BC decision-only: `artifacts/checkpoints/bc/phase2_bc_decisions.zip`
- Primo confronto fisso: `artifacts/evaluations/phase2_ppo_compare_fixed_30ep_seed42/`
- BC asincrono bimanuale: `artifacts/checkpoints/bc/phase2_bc_async_bimanual_recovery.zip`
- Ranking fully-minimal 1,5M: `artifacts/evaluations/phase2_async_bimanual_minimal_resume_1p5m_30ep/`
- Checkpoint full-task selezionato: `artifacts/evaluations/phase2_native_minimal_full_probe_50ep/phase2_native_minimal_full_best_50ep.zip`
- Ranking della nuova pipeline: `artifacts/evaluations/phase2_native_minimal_full_probe_50ep/`
- Evaluation recovery/full: `artifacts/evaluations/phase2_native_minimal_recovery_probe/`
- Evaluation multi-seed 150 episodi: `artifacts/evaluations/phase2_native_minimal_multiseed_150ep/`
- Job completato: `artifacts/jobs/phase2_native_minimal_phase1_recipe_pipeline/status.txt`
- Dataset teacher-terminal train: `artifacts/datasets/bc/phase2_phase1_teacher_terminal128_train_seed20261019.npz`
- Dataset teacher-terminal validation: `artifacts/datasets/bc/phase2_phase1_teacher_terminal32_valid_seed20261019.npz`
- BC teacher-terminal: `artifacts/checkpoints/bc/phase2_bc_phase1_teacher_terminal128_80ep_seed42.zip`
- Evaluation BC vero handoff 50 episodi: `artifacts/evaluations/phase12_teacherterminal_bc80_50ep_seed20261021/`
- Report PPO conservativo 400k: `artifacts/reports/random_sb3_log_sim_phase_2_phase1_teacherterminal_replay_kl_ppo_probe400k/training_analysis_report.png`
- Stato PPO conservativo: `artifacts/jobs/phase2_teacherterminal_ppo400k/status.txt`
- Collector stati di recovery: `scripts/collect_phase2_recovery_state_bank.py`
- Pipeline curriculum recovery: `scripts/run_phase2_recovery_curriculum_pipeline.sh`
- Stato curriculum recovery: `artifacts/jobs/phase2_recovery_curriculum500k/status.txt`
- Banca recovery: `artifacts/datasets/recovery/phase2_bc_recovery_states64_seed20261021.npz`
- Ranking recovery completo: `artifacts/evaluations/phase2_recovery_curriculum500k_bank64_rank_retry/checkpoint_ranking.csv`
- Confronto nominale diretto BC/PPO: `artifacts/evaluations/phase2_recovery_curriculum500k_nominal_direct50/checkpoint_ranking.csv`
- Report PPO recovery 500k: `artifacts/reports/random_sb3_log_sim_phase_2_teacherterminal_recoverybank50_replay003_kl001_ppo500k/training_analysis_report.png`

### Phase 3

- Dataset TCC: `/mnt/data/aiprah/data/random_sim_dataset_tcc/{train,valid}/phase_3/`
- Run TCC: `/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase3_tcc/`
- Dataset BC bimanuale: `artifacts/datasets/bc/phase3_bimanual_minimal160.npz`
- BC bimanuale 80 epoch: `artifacts/checkpoints/bc/phase3_bc_bimanual_minimal128_80ep_seed42.zip`
- PPO finale selezionato: `artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_3_bimanual_minimal_bc80_replay_probe_400k.zip`
- Copia del migliore: `artifacts/evaluations/phase3_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/phase3_bimanual_minimal_best_50ep.zip`
- Ranking deterministico: `artifacts/evaluations/phase3_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/checkpoint_ranking.csv`
- Report training: `artifacts/reports/random_sb3_log_sim_phase_3_bimanual_minimal_bc80_replay_probe_400k/training_analysis_report.png`
- Stato pipeline: `artifacts/jobs/phase3_bimanual_minimal_80ep_pipeline/status.txt`

### Phase 4

- Dataset TCC: `/mnt/data/aiprah/data/random_sim_dataset_tcc/{train,valid}/phase_4/`
- Run TCC: `/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase4_tcc/`
- Dataset BC bimanuale: `artifacts/datasets/bc/phase4_bimanual_minimal160.npz`
- BC bimanuale 80 epoch: `artifacts/checkpoints/bc/phase4_bc_bimanual_minimal128_80ep_seed42.zip`
- Checkpoint canonico selezionato: `artifacts/evaluations/phase4_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/phase4_bimanual_minimal_best_50ep.zip`
- Ranking canonico: `artifacts/evaluations/phase4_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/checkpoint_ranking.csv`
- Report canonico: `artifacts/reports/random_sb3_log_sim_phase_4_bimanual_minimal_bc80_replay_probe_400k/training_analysis_report.png`
- Report curriculum posa: `artifacts/reports/random_sb3_log_sim_phase_4_pose_curriculum_replay_800000/training_analysis_report.png`
- Ranking parked dopo curriculum: `artifacts/evaluations/phase4_pose_curriculum_replay_800000/parked_rank_50ep/checkpoint_ranking.csv`
- Script curriculum: `scripts/run_phase4_pose_curriculum.sh`
- Dataset teacher-terminal: `artifacts/datasets/bc/phase4_phase3_teacher_terminal160_seed20261026_demos.npz`
- Banca di stati fisici Phase 3: `artifacts/datasets/bc/phase4_phase3_teacher_terminal160_seed20261026_bank.npz`
- Nuovo BC teacher-terminal: `artifacts/checkpoints/bc/phase4_bc_phase3_teacher_terminal128_80ep_seed42.zip`
- Evaluation deterministica 20/20 Phase 3→4: `artifacts/evaluations/phase34_true_handoff_teacherterminal_bc80_20ep_seed20261027/`
- Pipeline BC ed evaluation: `scripts/run_phase4_teacher_terminal_bc_pipeline.sh`
- Collector teacher-terminal: `scripts/run_phase4_phase3_teacher_terminal_collection.sh`
- Coda di robustezza multi-seed/full-task: `scripts/run_phase4_teacherterminal_robustness_queue.sh`

### Confronto visuale e protocollo controllato

- Video minimal delle tre fasi: `artifacts/evaluations/minimal_mask_phase_inference_videos_20260927/`
- Pipeline controllata: `scripts/run_controlled_serial_minimal_protocol.sh`
- Stato pipeline: `artifacts/jobs/controlled_serial_minimal_1m_3seed/status.txt`
- Manifest sperimentale: `artifacts/jobs/controlled_serial_minimal_1m_3seed/protocol_manifest.txt`
- Evaluation: `artifacts/evaluations/controlled_serial_minimal_1m_3seed/`
- Report aggregato: `artifacts/reports/controlled_serial_minimal_1m_3seed/`
- Aggregatore: `evaluation/controlled_protocol_analysis.py`

### Full task e router

- Script full task: `scripts/eval_full_task.py`
- Router: `source/m_dVrk/m_dVrk/hrl/phase_router.py`
- Analisi router: `evaluation/phase_router_analysis.py`
- Evaluation router: `scripts/evaluate_phase_router.py`
- Metriche router first-frame ripulito: `artifacts/evaluations/phase_router_phase012_clean_first_frame_v2/metrics.json`
- Smoke full-task con transition parking: `artifacts/evaluations/full_task_phase012_seed42_parked_smoke/`
- Smoke full-task random 0,75: `artifacts/evaluations/full_task_phase012_random075_seed1337/` e `full_task_phase012_random075_seed2025/`
- Confronto success simbolico/TCC: `artifacts/evaluations/full_task_phase012_symbolic_trace_fixed_seed42/` e `full_task_phase012_tcc_success_fixed_seed42/`
- Router Phase 0–4: `artifacts/evaluations/phase_router_phase01234_first_frame_20261007/`
- Full task 0–4 randomizzazione 0,75: `artifacts/evaluations/full_task_phase01234_tcc_home_phase4_random075_seed42_20261008/`
- Full task 0–4 fisso con homing: `artifacts/evaluations/full_task_phase01234_tcc_home_phase4_fixed_seed42_20261008/`
- Script riproducibile: `scripts/run_full_task_phase01234_home_probe.sh`
- Evaluation Phase 3→4 multi-seed 100/100: `artifacts/evaluations/phase34_teacherterminal_bc80_multiseed100/`
- Summary aggregato Phase 3→4: `artifacts/jobs/phase4_teacherterminal_robustness_queue/phase34_multiseed_summary.json`
- Baseline full task TCC random 0,75, 5/20: `artifacts/evaluations/full_task_phase01234_teacherterminal_bc80_tcc_random075_20ep_seed20261033/`
- Full task con Phase 2 teacher-terminal PPO, 19/20: `artifacts/evaluations/full_task_phase01234_phase2_teacherterminal_ppo400k_tcc_random075_20ep_seed20261033/`
- Script del confronto full task: `scripts/run_full_task_phase01234_phase2_teacherterminal_ppo400k_tcc20.sh`
- Stato del confronto full task: `artifacts/jobs/full_task_phase01234_phase2_teacherterminal_ppo400k_tcc20/status.txt`
- Indice di report, video e comandi: `RISULTATI_E_COMANDI_INFERENZA.md`

---

## 6. Commit che scandiscono la storia

| Data | Commit | Significato |
|---|---|---|
| 17 mar | `eb4064f` | prima versione della repository |
| 24 apr | `8bf7087` | state-machine runner |
| 27 apr | `91d87c7` | asset dVRK/board e runner esteso |
| 28 apr | `28074a8` | terminazioni Isaac Lab, reset e parallel runner |
| 5 giu | `3dfd0dc` | HFSM, dataset TCC, evaluation e video |
| 29 lug | `fab2d32` | Phase 0 conclusa; BC e avvio Phase 1 |
| 6 ago | `248d653` | centralizzazione artifact/config e modularizzazione |
| 6 ago | `f02c9f9` | merge della centralizzazione in master |
| 1 set | `15a6f46` | revisione Phase 0 |
| 2 set | `47baad9` | controlli deterministici PPO e ranking |
| 3 set | `983b023` | ripristino bonus di successo |
| 3 set | `240a71e` | reach bounded e workspace per braccio |
| 4 set | `4708c2d` | curriculum randomizzazione success-gated |
| 5 set | `afc1dd4` | esperimento phase router TCC |
| 5 set | `abbcfb8` | start prototype dal primo frame |
| 5 set | `039221d` | workflow completo Phase 2 |
| 6 set | `a83a019` | correzione masking deterministico Phase 2 |

Dopo `a83a019`, la maggior parte dell'evoluzione descritta sopra vive ancora come modifiche non committate e artifact sul server. Prima di una pubblicazione o consegna della tesi, questo stato dovrebbe essere consolidato in commit tematici.

---

## 7. Cosa non è possibile affermare con certezza

- L'esatto ordine di tutti i tentativi tra marzo e maggio: la storia Git contiene pochi snapshot.
- La data in cui ogni idea è stata formulata: possiamo datare l'implementazione o l'esperimento, non sempre la discussione che lo ha preceduto.
- Una classifica unica tra esperimenti con maschere, seed o stati iniziali diversi.
- Che il 97/100 aggregato di Phase 1 equivalga a 100 prove indipendenti: i due run principali condividono il seed.
- Che una percentuale di training equivalga alla performance deterministica: curriculum e soft mask possono rendere il training più facile dell'evaluation finale fully-minimal.
- Che la Phase 2 fixed al 100% e la Phase 2 minimal al 46,7% siano in contraddizione: misurano problemi decisionali diversi.
- Che la differenza di bimanualità tra Phase 0 e Phase 1 sia causata soltanto dal task: le policy correnti hanno budget, replay e distribuzioni BC differenti. È precisamente la domanda del protocollo controllato avviato il 27 settembre.
- Che `127/150 = 84,7%` per Phase 2 misuri la variabilità tra training indipendenti: i cinque seed riguardano l'evaluation dello stesso checkpoint.
- Che il 96% di Phase 3 dimostri un vantaggio di PPO sul BC: nel ranking corrente entrambi ottengono 48/50 con metriche comportamentali quasi identiche.
- Che il 96% di Phase 3 misuri già la robustezza rispetto al seed di training: serve ancora ripetere BC/PPO con seed indipendenti e valutarli sullo stesso protocollo.
- Che il 100/100 Phase 3→4 equivalga a robustezza universale: copre cinque seed di evaluation ma una sola coppia di checkpoint e la distribuzione teacher-terminal simulata.
- Che il 20/20 validi il success detector TCC: in quella prova TCC seleziona correttamente la fase, ma il completamento viene dichiarato dalla condizione simbolica.
- Che il fallimento full-task randomizzato sia attribuibile soltanto a Phase 4: nel probe a scala `0,75` l'episodio si interrompe già in Phase 2.
- Che un router corretto equivalga a una composizione corretta delle policy: il probe fisso ottiene 4/4 transizioni corrette ma fallisce comunque durante l'esecuzione di Phase 4.
- Che l'homing cartesiano riproduca lo stato canonico della rete: posizione dell'end-effector, orientamento e configurazione articolare possono corrispondere a soluzioni IK differenti.
- Che il passaggio full-task dal 25% al 95% dimostri un vantaggio causale di PPO sul BC Phase 2: il PPO teacher-terminal e il BC hanno prestazioni standalone quasi identiche; il confronto valida soprattutto la nuova distribuzione teacher-terminal.
- Che 19/20 costituisca già una stima definitiva della robustezza end-to-end: è un singolo run da 20 episodi e richiede conferma multi-seed.

---

## 8. Come mantenere aggiornata la timeline

Per ogni nuovo esperimento importante aggiungere una voce contenente:

```text
Data:
Domanda sperimentale:
Dataset e checkpoint iniziale:
Maschera / curriculum:
Iperparametri distintivi:
Evaluation deterministica:
Esito e decisione conseguente:
Percorsi di log, report e checkpoint:
```

La timeline dovrebbe documentare non solo il training migliore, ma anche il motivo per cui un ramo è stato abbandonato. In questo progetto gli esperimenti negativi — collasso dopo l'unfreeze, fallimento guided→minimal, replay non stratificato — sono precisamente ciò che ha portato alla pipeline corrente.
