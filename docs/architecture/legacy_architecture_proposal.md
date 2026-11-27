# Proposta Di Struttura Del Progetto m_dVrk

Questo documento propone una riorganizzazione del progetto per separare meglio:

- codice IsaacLab dell'ambiente;
- logica HRL/SB3;
- raccolta dataset;
- pretraining BC;
- evaluation;
- artefatti sperimentali.

L'obiettivo principale e' evitare mismatch tra training, demo collection ed evaluation, specialmente su osservazioni, action space, success logic e wrapper.

## Diagnosi Attuale

La struttura IsaacLab di base e' sensata:

```text
source/m_dVrk/m_dVrk/tasks/manager_based/m_dvrk/
  m_dvrk_env_cfg.py
  mdp/
    events.py
    rewards.py
    terminations.py

source/m_dVrk/m_dVrk/assets/
  psm_cfg.py
```

Il problema e' che `scripts/` contiene ormai molta logica core:

```text
scripts/parallel_run.py        # training PPO + wrapper + reward + callbacks + eval/debug
scripts/parallel_env.py        # state machine + ring sync + high-level execution
scripts/collect_ppo_demos.py   # demo collection + wrapper duplicato
scripts/eval_run.py            # eval + wrapper duplicato
scripts/pretrain_ppo.py        # BC pretraining
scripts/check_bc_policy.py     # diagnostica BC
```

Questi file non sono piu' solo entrypoint. Contengono logica condivisa che deve rimanere identica tra:

- raccolta demo;
- pretraining BC;
- training PPO;
- evaluation;
- diagnostica offline.

Il bug su `manipulable_rings` e' un esempio diretto del rischio: la stessa osservazione era duplicata in piu' script.

## Struttura Target

Proposta:

```text
m_dVrk/
  ARCHITECTURE_PROPOSAL.md
  README.md
  pyproject.toml

  source/
    m_dVrk/
      m_dVrk/
        assets/
          __init__.py
          psm_cfg.py

        data/
          usd/

        hrl/
          __init__.py
          constants.py
          observations.py
          success.py
          rewards.py
          wrapper.py
          callbacks.py
          tcc.py
          bc.py
          config.py

        controllers/
          __init__.py
          spartan_state_machine.py
          ring_sync.py

        tasks/
          manager_based/
            m_dvrk/
              __init__.py
              m_dvrk_env_cfg.py
              agents/
                __init__.py
                rsl_rl_ppo_cfg.py
              mdp/
                __init__.py
                events.py
                rewards.py
                terminations.py

  scripts/
    train_ppo.py
    collect_ppo_demos.py
    pretrain_bc.py
    eval_policy.py
    check_bc_policy.py
    collect_tcc_dataset.py

  evaluation/
    plot_metrics.py
    plot_reward_progress.py
    training_analysis.py

  configs/
    ppo_phase0.yaml
    ppo_phase1.yaml
    bc_phase0.yaml
    bc_phase1.yaml
    collection_phase0.yaml
    collection_phase1.yaml

  artifacts/
    datasets/
      bc/
      tcc/
    checkpoints/
      bc/
      ppo/
      tcc/
    logs/
      bc/
      ppo/
      collection/
      eval/
    videos/
    snapshots/
    reports/
```

## Responsabilita Dei Moduli

### `source/m_dVrk/m_dVrk/hrl/constants.py`

Contiene una sola sorgente di verita' per action space e nomi semantici.

```python
VERB_MAP = {
    0: "reach",
    1: "grasp",
    2: "release",
    3: "idle",
}

TARGET_MAP = {
    0: "ring_red",
    1: "ring_yellow",
    2: "ring_green",
    3: "ring_blue",
    4: "peg_red",
    5: "peg_yellow",
    6: "peg_green",
    7: "peg_blue",
    8: "peg_gray",
    9: "None",
}

RING_NAMES = ["ring_red", "ring_yellow", "ring_green", "ring_blue"]
PEG_NAMES = ["peg_red", "peg_yellow", "peg_green", "peg_blue"]
```

Da qui devono importare tutti:

- PPO training;
- demo collection;
- evaluation;
- BC pretrain;
- offline diagnostics.

### `source/m_dVrk/m_dVrk/hrl/observations.py`

Contiene la logica condivisa per costruire osservazioni HRL:

- embedding stack;
- auxiliary observations;
- geometric observations;
- `manipulable_rings`;
- dimensioni attese.

Esempio di layout osservazione:

```text
obs = [
  emb_stack,       # 3 * 32 = 96
  aux_obs,         # 62
  geom_obs,        # 86
]

total = 244
```

Questo modulo deve essere usato da:

- `DVRKVisionHRLWrapper`;
- collector demo;
- eval;
- eventuali test/diagnostiche.

Regola: se cambia l'osservazione, deve cambiare in un solo punto.

### `source/m_dVrk/m_dVrk/hrl/success.py`

Contiene la logica di successo e progress count:

```text
phase_0:
  4 ring su peg_green

phase_1:
  2 ring su peg_red
  2 ring su peg_blue
  tutti i ring frozen
```

Qui dovrebbero vivere:

- `_peg_ring_count`;
- `_task_success`;
- eventuale `rings_complete`;
- eventuale progress score per reward shaping.

### `source/m_dVrk/m_dVrk/hrl/rewards.py`

Contiene reward usato da PPO/SB3.

Stato attuale:

```python
reward = -tcc_distance * 1e-3
reward -= invalid_commands * 1.0
```

Proposta futura:

```text
reward =
  TCC distance reward
  + phase progress shaping
  - invalid command penalty
  - overfill penalty
  + terminal success bonus
```

Per phase1:

```text
progress_score = min(red_count, 2) + min(blue_count, 2)
overfill = max(red_count - 2, 0) + max(blue_count - 2, 0)
```

### `source/m_dVrk/m_dVrk/hrl/wrapper.py`

Contiene `DVRKVisionHRLWrapper`, cioe' il VecEnv SB3 che:

- prende l'ambiente IsaacLab;
- mantiene embedding buffer;
- costruisce osservazioni;
- riceve azioni discrete HRL;
- sanitizza comandi;
- chiama la state machine;
- calcola reward;
- gestisce reset/done/info;
- espone `observation_space` e `action_space`.

Questo e' il punto piu' importante da centralizzare.

Obiettivo: training, collection ed evaluation devono usare lo stesso wrapper, non copie diverse.

### `source/m_dVrk/m_dVrk/hrl/callbacks.py`

Contiene callback SB3:

- `EpisodeCsvLoggerCallback`;
- `BestTerminalDistanceCheckpointCallback`;
- `FreezePolicyCallback`;
- eventuale callback futura per KL-to-BC o BC auxiliary loss.

### `source/m_dVrk/m_dVrk/hrl/tcc.py`

Contiene:

- `XIRLResnet18`;
- caricamento checkpoint TCC;
- `resolve_tcc_checkpoint`;
- preprocessing immagini;
- `compute_average_goal_embedding`.

Questo evita di duplicare la rete TCC in training, collector ed eval.

### `source/m_dVrk/m_dVrk/hrl/bc.py`

Contiene utilities per behavior cloning:

- caricamento dataset NPZ;
- validazione `obs_dim`;
- splitting logits MultiDiscrete;
- loss CE sulle quattro action heads;
- accuracy per head;
- diagnostiche semantic-invalid.

`scripts/pretrain_bc.py` dovrebbe solo chiamare questo modulo.

### `source/m_dVrk/m_dVrk/hrl/config.py`

Contiene dataclass di configurazione comuni:

```python
@dataclass
class PPOConfig:
    learning_rate: float
    ent_coef: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    vf_coef: float
    max_grad_norm: float
```

Questo aiuta a evitare parametri hardcoded dentro gli script.

### `source/m_dVrk/m_dVrk/controllers/spartan_state_machine.py`

Sposta qui la state machine oggi in `scripts/parallel_env.py`.

Contiene:

- high-level state machine;
- reach/grasp/release;
- peg inventory;
- attached/frozen state;
- stack level;
- top ring logic;
- reset phase0/phase1.

### `source/m_dVrk/m_dVrk/controllers/ring_sync.py`

Contiene la logica di sincronizzazione fisica dei ring:

- attached ring segue tip;
- frozen ring resta su peg;
- scrittura root state in Isaac;
- reset sync.

Questa logica oggi e' troppo intrecciata con wrapper/training.

## Scripts Target

Gli script devono diventare entrypoint sottili.

### `scripts/train_ppo.py`

Responsabilita':

- parsing CLI;
- setup IsaacLab `AppLauncher`;
- costruzione env;
- costruzione wrapper;
- caricamento TCC;
- caricamento checkpoint BC opzionale;
- creazione PPO;
- avvio `model.learn`.

Non dovrebbe contenere:

- definizione osservazioni;
- state machine;
- reward logic lunga;
- callback class;
- TCC model class.

### `scripts/collect_ppo_demos.py`

Responsabilita':

- parsing CLI;
- setup IsaacLab;
- avvio expert/state machine;
- raccolta transizioni;
- salvataggio NPZ.

Deve usare lo stesso wrapper e le stesse osservazioni del training.

### `scripts/pretrain_bc.py`

Responsabilita':

- caricare dataset;
- creare dummy env con `obs_dim`;
- allenare policy PPO via CE;
- salvare checkpoint.

La logica di BC dovrebbe stare in `hrl/bc.py`.

### `scripts/eval_policy.py`

Responsabilita':

- caricare modello;
- caricare VecNormalize se serve;
- eseguire eval;
- salvare video/report.

Deve usare lo stesso wrapper del training.

### `scripts/check_bc_policy.py`

Responsabilita':

- diagnostica offline;
- confronto raw vs normalized obs;
- accuracy dataset;
- semantic invalid actions;
- sanity check su feature geometriche.

## Configurazioni

I parametri non dovrebbero vivere solo come costanti hardcoded.

Proposta:

```text
configs/
  ppo_phase0.yaml
  ppo_phase1.yaml
  bc_phase0.yaml
  bc_phase1.yaml
  collection_phase0.yaml
  collection_phase1.yaml
```

Esempio `configs/ppo_phase1.yaml`:

```yaml
task_phase: phase_1
num_envs: 64
total_timesteps: 5000000
freeze_policy_timesteps: 500000

vecnormalize:
  norm_obs: false
  norm_reward: true
  clip_obs: inf
  clip_reward: inf

ppo:
  learning_rate: 0.00003
  n_steps: 128
  batch_size: 2048
  n_epochs: 3
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.1
  ent_coef: 0.001
  vf_coef: 0.5
  max_grad_norm: 0.5
  target_kl: 0.01

policy:
  net_arch:
    pi: [64, 64]
    vf: [64, 64]
  activation: relu

paths:
  goal_dataset_root: /mnt/data/aiprah/data/random_sim_dataset_tcc
  pretrained_checkpoint: /home/aiprah/Documents/m_dVrk/artifacts/checkpoints/bc/randomized_dvrk_ppo_pretrained_phase1.zip
  output_dir: /home/aiprah/Documents/m_dVrk/artifacts/logs/ppo
```

## Artefatti

La root del repo oggi contiene molti artefatti. Proposta:

```text
artifacts/
  datasets/
    bc/
      dataset_supervisionato_randomized_phase0.npz
      dataset_supervisionato_randomized_phase1.npz
    tcc/

  checkpoints/
    bc/
      randomized_dvrk_ppo_pretrained_phase0.zip
      randomized_dvrk_ppo_pretrained_phase1.zip
    ppo/
      phase0/
      phase1/
    tcc/

  logs/
    bc/
    collection/
    ppo/
      phase0/
      phase1/
    eval/

  videos/
    collection/
    eval/
    success/

  snapshots/
    reward_debug/
    success/

  reports/
    plots/
    markdown/
```

Questi artefatti dovrebbero essere ignorati da git, salvo piccoli report scelti manualmente.

## `.gitignore` Consigliato

```gitignore
# Python
__pycache__/
*.pyc
.venv/

# Isaac/colcon output
log/
install/
build/

# Training artifacts
artifacts/datasets/
artifacts/checkpoints/
artifacts/logs/
artifacts/videos/
artifacts/snapshots/

# Legacy local artifacts
random_sb3_log*/
sb3_log*/
modelli_salvati*/
success_snapshots/
success_videos/
video_raccolti/
reward_debug_frames/
*.log
*.npz
*.mp4
*.png

# Local raw data dumps
${data}/
Data/
```

## Piano Di Migrazione Graduale

### Fase 1: Stabilizzare Senza Cambiare Comportamento

Obiettivo: nessun cambiamento algoritmico.

1. Creare `hrl/constants.py`.
2. Far importare `VERB_MAP`, `TARGET_MAP`, `RING_NAMES`, `PEG_NAMES` da li'.
3. Creare `hrl/tcc.py`.
4. Spostare `XIRLResnet18` e caricamento TCC.
5. Verificare che `parallel_run.py`, `collect_ppo_demos.py`, `eval_run.py` compilino.

### Fase 2: Centralizzare Osservazioni

Obiettivo: una sola sorgente di verita' per `obs_dim` e feature order.

1. Creare `hrl/observations.py`.
2. Spostare `aux_obs`, `geom_obs`, `manipulable_rings`.
3. Aggiungere assert sulle dimensioni:

```text
emb_dim = 96
aux_dim = 62
geom_dim = 86
obs_dim = 244
```

4. Far usare la stessa logica a train, collect, eval.
5. Rigenerare un piccolo dataset demo e controllare:

```text
obs.shape[1] == 244
manipulable_rings non sempre zero
azioni grasp/reach target ring coerenti con mask
```

### Fase 3: Centralizzare Wrapper

Obiettivo: un solo `DVRKVisionHRLWrapper`.

1. Creare `hrl/wrapper.py`.
2. Spostare la classe wrapper da `parallel_run.py`.
3. Usarla in:
   - `train_ppo.py`;
   - `collect_ppo_demos.py`;
   - `eval_policy.py`.

### Fase 4: Spostare State Machine

Obiettivo: togliere logica core da `scripts/parallel_env.py`.

1. Creare `controllers/spartan_state_machine.py`.
2. Creare `controllers/ring_sync.py`.
3. Lasciare eventualmente `scripts/parallel_env.py` come shim temporaneo.

### Fase 5: Config E Artefatti

Obiettivo: training riproducibili e ordinati.

1. Creare `configs/*.yaml`.
2. Rendere CLI compatibile con config file.
3. Spostare output in `artifacts/`.
4. Aggiungere naming run standard:

```text
phase1_bcraw_lr3e-5_ent0.001_freeze500k_seed42_YYYYMMDD_HHMM
```

## Regole Pratiche Per Il Futuro

1. Le osservazioni devono essere definite in un solo modulo.
2. Il collector deve usare lo stesso wrapper del training, salvo differenze esplicite e documentate.
3. Ogni dataset BC deve salvare anche metadata:

```text
obs_dim
geom_dim
aux_dim
feature_version
task_phase
git_commit
command
timestamp
```

4. Ogni checkpoint BC deve avere accanto un file metadata:

```text
dataset_path
dataset_hash
obs_dim
feature_version
epochs
lr
final_loss
head_accuracies
```

5. Ogni run PPO deve salvare:

```text
config.yaml
command.txt
git_status.txt
progress.csv
episode_progress.csv
vecnormalize.pkl
```

6. Non riusare directory di log senza suffix esplicito.
7. Non confrontare run se non hanno:

```text
stesso checkpoint BC
stesso dataset BC
stessa feature_version
stesso reward_version
stessa success_version
```

## Priorita Consigliata

La priorita' piu' alta e':

```text
1. hrl/constants.py
2. hrl/observations.py
3. hrl/wrapper.py
4. metadata dataset/checkpoint
5. artifacts/ ordinato
```

Solo dopo conviene investire su cambi algoritmici piu' sofisticati, come:

- learning rate piu' basso dopo BC;
- `ent_coef` piu' basso;
- `target_kl`;
- reward shaping phase1;
- KL-to-BC;
- auxiliary BC loss durante PPO.

La ragione e' semplice: prima serve rendere affidabile il circuito sperimentale. Poi ha senso ottimizzare gli iperparametri.
