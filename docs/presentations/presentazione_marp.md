---
marp: true
theme: gaia
class: invert
paginate: true
backgroundColor: #1a1a24
color: #e0e0e0
headingDivider: 2
size: 16:9
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    padding: 40px;
  }
  h1 {
    color: #4da6ff;
  }
  h2 {
    color: #4da6ff;
    border-bottom: 2px solid #33334d;
  }
  footer {
    font-size: 0.5em;
    color: #888;
  }
  code {
    background-color: #2b2b36;
    color: #ff9d00;
  }
---

# Controllo Chirurgico Robotico Bimanuale con RL Ibrido
## Pianificazione gerarchica con PPO, SPARTAN e TCC su dVRK
[Tuo Nome] / Luglio 2026

---

## Obiettivo RL: Fare "Reasoning"
* **RL per il Decision-Making ad Alto Livello**
  * Spazio di azioni discretizzato (MultiDiscrete): `[verb_l, target_l, verb_r, target_r]`
  * L'agente decide *cosa* fare (es. `grasp ring_red` per braccio L, `reach peg_green` per braccio R).
* **Perché evitare l'RL continuo end-to-end?**
  * Spazio di azione continuo immenso (18+ DoF bimanuale).
  * Rischio di collisioni violente ed esplorazione inefficiente.
* **Macchina a Stati SPARTAN (Low-Level)**
  * Esegue i movimenti continui, la cinematica inversa (IK) e la gestione di precisione delle pinze.

---

## Task Definition: Peg and Ring (Why & Setup)
* **Perché il Peg and Ring?**
  * Task classico dell'addestramento chirurgico (FLS) per la coordinazione bimanuale fine.
  * Workspace condiviso ridotto $\rightarrow$ alto rischio di collisioni tra i bracci.
* **Il Setup in Simulazione**
  * Ambiente fisico ad alte prestazioni in **Isaac Lab**.
  * **Elementi**: 2 bracci dVRK (PSM), 4 anelli colorati (rosso, giallo, verde, blu), peg corrispondenti di partenza, peg target verde, e peg grigi ausiliari come ostacoli.

---

## Regole del Task e Randomizzazione
* **Le Regole del Gioco (Rules)**
  * *Goal*: Posizionare tutti e 4 gli anelli colorati sul **peg verde**.
  * *Top-Ring Constraint (LIFO)*: Si può afferrare un anello solo se è in cima alla pila.
  * *Grasp/Release*: Grasp attivo entro $1.5\text{ cm}$. Rilascio vicino a un peg impila l'anello (spessore $1\text{ mm}$), altrimenti cade sulla board.
* **L'importanza della Randomizzazione**
  * *Fixed Mode*: Anelli sempre ordinati $\rightarrow$ Overfitting delle coordinate spaziali.
  * *Randomized Mode* (`--randomize_rings`): Reset degli anelli con XY e yaw casuali.
  * *Generalizzazione*: Costringe l'RL ad usare feedback visivo (TCC) e vettori geometrici relativi.

---

## Implementazione dell'Agente: TCC e PPO
* **TCC (Temporal Cycle Consistency) come Reward**
  * Encoder ResNet18 mappa i frame in uno spazio latente di 32 dimensioni ($z_t$).
  * Osservazione visiva: Stack temporale di 3 frame ($96\text{ dim}$) per catturare la dinamica.
  * Reward denso visivo (distanza euclidea negativa dal goal):
    $$r_t = -\|z_t - z_{goal}\|^2 \cdot 10^{-3}$$
* **Transizione da A2C a PPO**
  * *A2C*: Instabilità e alta varianza negli aggiornamenti della policy discreta.
  * *PPO*: Stabilità grazie al clipping dell'obiettivo. Convergenza più rapida (1.5M vs 4.6M steps di A2C).

---

## Accorgimenti Tecnici e Critic Freezing
* **Risoluzione delle Sfide Fisiche (Failure Modes)**
  * *Latenza Visiva*: Risolta con **5 Settle Steps** di attesa post-task prima del reset (permette l'allineamento del rendering della camera).
  * *Invalid Action Spamming*: Penalità di $-1.0$ nel reward per comandi non coerenti.
* **Pre-train Policy & Critic Freezing**
  * Caricare una policy pre-addestrata (Behavior Cloning) con un Critic casuale causa aggiornamenti distruttivi di PPO.
  * *Soluzione*: Congelare i pesi dell'Actor (`policy_net`/`action_net`) per il primo $1/5$ di passi (es. 150k step) per fare il **warm-up del Critic**, scongelando poi tutto per il fine-tuning.

---

## Risultati Ottenuti e Conclusioni
* **Risultati Correnti (Today Results)**
  * Minimizzazione della distanza latente (fino a media di $\sim 2200$).
  * Coordinazione bimanuale fluida (parallelizzazione dei task o attesa coordinata).
  * Robustezza totale alla randomizzazione di anelli e orientamenti.
* **Conclusioni**
  * L'approccio RL Ibrido + SPARTAN rende fattibili compiti bimanuali chirurgici complessi.
  * TCC fornisce reward visivi densi efficaci senza sensori fisici privilegiati.
  * *Prossimo passo*: Trasferimento Sim-to-Real sui bracci dVRK reali in laboratorio.
