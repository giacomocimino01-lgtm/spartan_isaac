# Presentazione: Apprendimento per Rinforzo Ibrido per il Task Peg-and-Ring su dVRK

Questo documento contiene la scaletta dettagliata slide-per-slide, i messaggi chiave, i contenuti tecnici (formule, parametri, logica) e i suggerimenti di layout visivo per preparare la presentazione PowerPoint.

---

## Slide 1: Titolo e Introduzione
*   **Titolo Principale**: Controllo Chirurgico Robotico Bimanuale con RL Ibrido
*   **Sottotitolo**: Pianificazione gerarchica ad alto livello (PPO), controllo motorio discreto (SPARTAN) e reward visivi auto-supervisionati (TCC) su dVRK
*   **Relatore**: [Tuo Nome]
*   **Messaggio Chiave**: Risolvere compiti complessi di manipolazione bimanuale fine separando la pianificazione strategica (RL) dal controllo esecutivo continuo (macchina a stati), guidati da un reward puramente visivo.
*   **Layout Visivo Consigliato**:
    *   Sfondo scuro ed elegante (tonalità blu scuro/grigio antracite).
    *   Immagine o screenshot della simulazione bimanuale con i due bracci dVRK (Left & Right Arm), gli anelli colorati e il peg verde al centro in Isaac Lab.

---

## Slide 2: Obiettivo RL: Fare "Reasoning" e Pianificazione
*   **Titolo**: Reinforcement Learning per il Decision-Making ad Alto Livello
*   **Messaggio Chiave**: L'RL non deve imparare il controllo motorio continuo da zero, ma deve focalizzarsi sulla sequenzialità logica del task.
*   **Perché l'RL continuo end-to-end fallisce**:
    *   **Alta dimensionalità**: Più di 18 gradi di libertà (DoF) da controllare simultaneamente per due bracci chirurgici.
    *   **Reward sparse**: Imparare a infilare un anello muovendo i motori millimetricamente richiede miliardi di campioni ed è prono a fallimenti locali.
    *   **Collisioni**: L'esplorazione casuale porta a contatti violenti tra i bracci robotici.
*   **La soluzione Ibrida**:
    *   **RL come Pianificatore (High-Level)**: Opera in uno spazio di azioni discretizzato (MultiDiscrete: `[verb_l, target_l, verb_r, target_r]`). L'agente decide *cosa* fare (es. `grasp ring_red` per il braccio sinistro, `reach peg_green` per il destro).
    *   **Macchina a Stati SPARTAN (Low-Level)**: Riceve i comandi discreti e si occupa di calcolare la cinematica inversa (IK), le traiettorie lineari di discesa/salita, la gestione delle pinze e il sollevamento sicuro del braccio.
*   **Layout Visivo Consigliato**:
    *   Schema a blocchi gerarchico:
        `Osservazione (Stato Visivo + Fisico) ➔ Agente RL (PPO) ➔ Azione Discreta (Verbo/Target) ➔ Macchina a Stati (SPARTAN) ➔ Azioni Continue (Giunti/Pinze) ➔ Simulatore (Isaac Lab)`.

---

## Slide 3: Task Definition: Peg and Ring (1/2)
*   **Titolo**: Task Definition: Il Benchmark "Peg and Ring"
*   **Messaggio Chiave**: Un classico compito di chirurgia robotica (FLS) importato in simulazione parallela per testare bimanualità e coordinazione.
*   **3.1 Why Peg and Ring?**:
    *   Richiede un elevato grado di coordinazione spaziale fine tra i due bracci (workspace condiviso ristretto).
    *   È un compito asimmetrico: un braccio può dover stabilizzare o liberare spazio mentre l'altro inserisce, oppure passarsi gli oggetti.
    *   Evidenzia i problemi tipici del controllo rigido di precisione.
*   **3.2 Our Peg and Ring Setup**:
    *   **Ambiente**: Isaac Lab (NVIDIA Isaac Sim) con rendering GPU-accelerato.
    *   **Componenti**:
        *   2 bracci dVRK (Patient Side Manipulators - PSM).
        *   4 anelli colorati (rosso, giallo, verde, blu).
        *   Peg corrispondenti per il posizionamento iniziale, un peg verde di destinazione (*target*) e peg grigi ausiliari che fungono da ostacoli o appoggi temporanei.
*   **Layout Visivo Consigliato**:
    *   Screenshot ravvicinato del pegboard e degli anelli con didascalie che indicano i componenti (Rings, Pegs, Target Peg).

---

## Slide 4: Task Definition: Peg and Ring (2/2)
*   **Titolo**: Regole del Task e Randomizzazione
*   **Messaggio Chiave**: Regole strutturate per l'interazione ed esigenza di generalizzazione visiva.
*   **2.1 Rules (Le Regole di Gioco)**:
    *   **Goal**: Impilare tutti e 4 gli anelli colorati sul **peg verde**.
    *   **Vincolo Top-Ring (LIFO)**: È possibile afferrare un anello solo se è libero in cima alla pila del rispettivo peg. Non si possono estrarre anelli coperti da altri (logica di stacking).
    *   **Dinamica di Grasp/Release**:
        *   *Grasp*: Avviene se la pinza si chiude entro la tolleranza geometrica (`GRASP_ATTACH_DISTANCE` = 1.5 cm) vicino all'anello.
        *   *Release*: Se effettuato vicino a un peg entro la tolleranza (`RELEASE_PEG_CAPTURE_DISTANCE` = 1.5 cm), l'anello si allinea e si congela sul peg ad un'altezza proporzionale al numero di anelli già presenti (1 mm di spessore per anello). Se rilasciato altrove, cade sulla board.
*   **2.2 Randomization! Why is it important for generalization**:
    *   *Fixed Layout*: Gli anelli partono sempre in posizioni fisse. L'RL tende a memorizzare le coordinate assolute (overfitting).
    *   *Randomized Reset* (`--randomize_rings`): Ad ogni inizio episodio, gli anelli vengono spawnati con coordinate XY e angoli di yaw casuali.
    *   *Importanza*: Forza l'agente a fare affidamento esclusivamente sulla telecamera (embedding visivi) e sulla geometria relativa per localizzare gli anelli, garantendo la robustezza necessaria per il sim-to-real.
*   **Layout Visivo Consigliato**:
    *   Affiancamento:
        *   A sinistra: Immagine del layout fisso (perfettamente ordinato).
        *   A destra: Esempio di layout randomizzato (anelli sparsi e ruotati in disordine sulla board).

---

## Slide 5: Agent Implementation (1/2): TCC e Algoritmi
*   **Titolo**: Rappresentazione Visiva e Algoritmi di Apprendimento
*   **Messaggio Chiave**: TCC per guidare l'apprendimento visivo e PPO per ottimizzare stabilmente la politica.
*   **3.1 TCC (Temporal Cycle Consistency) come segnale di Reward**:
    *   Un encoder ResNet18 pre-addestrato mappa i frame della telecamera (ridimensionati a 112x112) in uno spazio latente di 32 dimensioni ($z_t$).
    *   L'osservazione dell'agente accumula uno stack temporale di 3 frame latenti ($96$ dimensioni totali) per catturare la velocità e la direzione del movimento.
    *   **Reward denso visivo**: Calcolato come la distanza euclidea negativa dal goal di successo registrato:
        $$r_t = -\|z_t - z_{goal}\|^2 \cdot 10^{-3}$$
    *   Nessun sensore fisico o distanza cartesiana ideale usata per calcolare il reward: l'RL impara guardando la scena.
*   **3.2 Transizione da A2C a PPO**:
    *   *Prima A2C (Advantage Actor-Critic)*: Test iniziali caratterizzati da alta varianza e instabilità negli aggiornamenti della policy discreta.
    *   *Poi PPO (Proximal Policy Optimization)*: Il clipping della funzione obiettivo ha impedito aggiornamenti di peso troppo drastici, stabilizzando la convergenza.
    *   *Efficienza*: PPO ha raggiunto la convergenza ed una bassa distanza latente in soli 1.5M di step, mentre A2C ha richiesto oltre 4M di step per risultati meno stabili.
*   **Layout Visivo Consigliato**:
    *   Grafico concettuale del funzionamento di TCC: frame video proiettati come punti in uno spazio 3D che si avvicinano al "Goal Embedding".

---

## Slide 6: Agent Implementation (2/2): Dettagli e Trucchi Tecnici
*   **Titolo**: Risoluzione delle Sfide Fisiche e Addestramento del Critic
*   **Messaggio Chiave**: Settle steps per contrastare la latenza visiva, penalità per comandi inutili e warm-up del Critic per preservare il Behavior Cloning.
*   **3.3 First Results & Failure Modes (Primi tentativi falliti)**:
    *   *Latenza Visiva del Simulatore*: Isaac Sim impiega alcuni frame a propagare graficamente i posizionamenti fisici degli anelli. L'RL resettava o agiva prima dell'aggiornamento grafico della telecamera.
        *   **Soluzione (Settle Steps)**: Introdotti 5 step di attesa (*settling steps*) dopo la logica di successo prima del reset o della convalida visiva.
    *   *Spam di azioni non valide*: L'agente continuava a richiedere grasp/release quando le braccia erano già impegnate in altri movimenti.
        *   **Soluzione (Invalid Command Penalty)**: Assegnata una penalizzazione di $-1.0$ ad ogni azione richiesta non valida, accelerando drasticamente l'apprendimento delle regole del robot.
*   **3.4 Pre-train Policy e Critic Freezing**:
    *   *Il problema*: Caricare una policy pre-addestrata con Behavior Cloning (BC) su dati di esperti in una rete RL con Critic (Value Network) non addestrato genera stime di vantaggio errate. I primi aggiornamenti di PPO distruggerebbero la policy del BC.
    *   *La soluzione*: Congelare i pesi dell'Actor (`policy_net` e `action_net`) per la prima fase dell'addestramento (1/5 degli step totali, es. primi 150k step).
    *   *Critic Warm-up*: Durante questo blocco, solo il Critic aggiorna i suoi pesi per imparare a stimare correttamente il valore degli stati della policy esperta. Dopo il congelamento, la policy viene sbloccata e ottimizzata stabilmente.
*   **Layout Visivo Consigliato**:
    *   Timeline grafica o schema temporale dell'addestramento:
        `[Step 0 - 150k]: Actor FREEZE (Solo Critic si allena)` ➔ `[Step 150k - 5M]: Actor & Critic UNFREEZE (Fine-tuning RL)`.

---

## Slide 7: Risultati Correnti (Today Results) e Conclusioni
*   **Titolo**: Risultati Ottenuti e Conclusioni
*   **Messaggio Chiave**: Convergenza robusta in ambienti randomizzati e coordinazione bimanuale efficace.
*   **3.5 Today Results**:
    *   **Successo di Convergenza**: Il modello PPO ha minimizzato la distanza latente dal goal in modo stabile ad ogni episodio (raggiungendo distanze medie eccellenti nell'ordine di ~2200).
    *   **Comportamento Bimanuale Coordinato**: I bracci dVRK dimostrano un comportamento fluido: un braccio aspetta o sgombra il pegboard mentre l'altro recupera e sposta gli anelli in parallelo.
    *   **Resilienza alla randomizzazione**: Il sistema gestisce con successo la dislocazione casuale di anelli e peg, senza fallimenti di grasping.
*   **Conclusioni**:
    *   L'architettura ibrida PPO + SPARTAN riduce drasticamente lo spazio di esplorazione rendendo fattibile il controllo bimanuale di precisione.
    *   Il Critic Freezing è essenziale quando si unisce Behavior Cloning e RL per evitare il degrado iniziale.
    *   TCC consente di creare reward densi stabili per la manipolazione chirurgica senza sensori fisici privilegiati.
*   **Layout Visivo Consigliato**:
    *   Grafico di convergenza del reward o della terminal distance (es. da `reward_progress.png` o `reward_per_step_progress.png`).
    *   Box finale per i "Prossimi Sviluppi: Test su dVRK reale (Sim-to-Real)".
