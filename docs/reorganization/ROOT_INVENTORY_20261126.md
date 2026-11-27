# Inventario della root prima della riorganizzazione

Data dell'inventario: 26 novembre 2026.

Checkout server analizzato: `/home/aiprah/Documents/m_dVrk` sul branch
`backup/pre-real-deploy-20261126-server`, commit `4d34c7d`.

Questo documento propone una classificazione. Non e' ancora stato spostato,
rinominato o cancellato alcun elemento della repository.

## Quadro generale

- 296 file direttamente nella root.
- 257 file di log, per circa 174 MiB.
- 29 directory direttamente nella root, escluse le normali entry implicite.
- `artifacts/` occupa circa 13 GiB ed e' gia' la destinazione corretta per gli
  output sperimentali.
- Le directory piu' grandi fuori da `artifacts/` sono `video_raccolti/`
  (9,9 GiB), `success_snapshots/` (1,7 GiB), `${data}/` (774 MiB),
  `success_videos/` (485 MiB) e `Data/` (257 MiB).

## 1. Da conservare nella root

Questi elementi definiscono la repository o sono punti di ingresso standard.

### File

- `.dockerignore`
- `.gitattributes`
- `.gitignore`
- `.pre-commit-config.yaml`
- `README.md`
- `CHANGELOG.md`
- `PR_BODY.md`
- `pyproject.toml`

### Directory

- `.git/`
- `.github/`
- `.vscode/`
- `source/`
- `scripts/`
- `evaluation/`
- `tests/`
- `configs/`
- `artifacts/`

`scripts/` ed `evaluation/` restano inizialmente al loro posto. La loro
classificazione interna sara' una fase successiva, coperta dai test, per non
rompere import, launcher e path usati dagli esperimenti.

## 2. Da spostare in `docs/`

| Elemento attuale | Destinazione proposta | Motivo |
|---|---|---|
| `ARCHITECTURE_PROPOSAL.md` | `docs/architecture/legacy_architecture_proposal.md` | Proposta storica, in parte gia' implementata |
| `PROJECT_TIMELINE.md` | `docs/history/PROJECT_TIMELINE.md` | Ricostruzione storica del progetto |
| `RISULTATI_E_COMANDI_INFERENZA.md` | `docs/results/RISULTATI_E_COMANDI_INFERENZA.md` | Indice di risultati e comandi |
| `presentazione_marp.md` | `docs/presentations/presentazione_marp.md` | Sorgente della presentazione |
| `presentazione_tcc_dvrk.md` | `docs/presentations/presentazione_tcc_dvrk.md` | Sorgente della presentazione |
| `2106.03911v3.pdf` | `docs/references/xirl_2106.03911v3.pdf` oppure archivio bibliografico esterno | Paper XIRL originale |

`presentazione_marp.html` e' invece un output generato e va negli artifact.

## 3. Da spostare in `artifacts/`

### Evaluation full-task del 13 novembre

I seguenti file hanno lo stesso timestamp e appartengono alla stessa run da un
episodio, calibrata, seed 42, terminata con fallimento:

- `summary.json`
- `episodes.csv`
- `phase_completions.csv`
- `transitions.csv`
- `tcc_distance_trace.csv`
- `transition_score_trace.csv`
- `phase_entry_observations.npz`
- `episode_000_fail.mp4`
- `episode_000_phase_0_transition.png`
- `episode_000_phase_1_transition.png`

Destinazione proposta:

```text
artifacts/evaluations/legacy_root/
  full_task_calibrated_1ep_seed42_20261113/
```

### Presentazioni e media selezionati

| Elemento | Destinazione proposta |
|---|---|
| `presentation_assets/` | `artifacts/presentations/supervisor_20261125/` |
| `presentazione_marp.html` | `artifacts/presentations/legacy/` |
| `pov_operazione.mp4` | `artifacts/videos/legacy_inference/` |
| `pov_operazione_phase_0.mp4` | `artifacts/videos/legacy_inference/` |
| `episode_000_fail_phase_0.mp4` | `artifacts/videos/legacy_inference/` |
| `episode_001_success_phase_0.mp4` | `artifacts/videos/legacy_inference/` |

### Output di training gia' strutturati ma nella root

| Elemento | Dimensione | Destinazione proposta |
|---|---:|---|
| `random_sb3_log_sim_phase_1_bc_raw_no_norm_300k/` | 1,2 MiB | `artifacts/logs/legacy/phase_1/` |
| `random_sb3_log_sim_phase_1_bc_raw_no_norm_lr3e-5_ent0.001_freeze500k_2m/` | 1,1 MiB | `artifacts/logs/legacy/phase_1/` |
| `random_sb3_log_sim_phase_1_bc_raw_no_norm_lr3e-5_ent0.001_freeze500k_5m/` | 3,0 MiB | `artifacts/logs/legacy/phase_1/` |
| `random_sb3_log_sim_phase_1_toy_bc_raw_no_norm_lr3e-5_ent0.001_freeze500k_5m/` | 16 KiB | `artifacts/logs/legacy/phase_1/` |
| `sb3_log/` | 356 KiB | `artifacts/logs/legacy/` |
| `tensorboard_logs/` | 44 KiB | `artifacts/logs/legacy/` |
| `reward_debug_frames/` | 172 KiB | `artifacts/snapshots/legacy_reward_debug/` |

## 4. Da archiviare come storico

### Log sciolti

Tutti i 257 file `*.log` della root sono output storici, non sorgenti. La
destinazione proposta e':

```text
artifacts/legacy_root/logs/
  training/
  collection/
  evaluation/
  ranking/
  dataset_checks/
  launchers/
  misc/
```

Distribuzione principale:

| Prefisso | File | Dimensione approssimata |
|---|---:|---:|
| `train_*` | 41 | 84,0 MiB |
| `collect_*` | 33 | 64,6 MiB |
| `check_*` | 31 | 0,1 MiB |
| `eval_*` | 24 | 10,9 MiB |
| `rank_*` | 22 | trascurabile |
| `phase2_*` | 18 | 0,5 MiB |
| `pretrain_*` | 14 | 0,3 MiB |
| altri prefissi | 74 | circa 13,4 MiB |

Anche `nohup.out`, attualmente tracciato per errore, e `log.txt` appartengono
a questa categoria. Prima dello spostamento `nohup.out` dovra' essere rimosso
dal tracking Git senza cancellarne il contenuto.

### Checkpoint e risultati delle prime run

| Directory | Contenuto | Dimensione | Destinazione proposta |
|---|---|---:|---|
| `res_1.5mln/` | Primi checkpoint A2C | 2,7 MiB | `artifacts/checkpoints/legacy/early_runs/` |
| `res_2.5mln/` | Checkpoint PPO, VecNormalize e curve | 7,0 MiB | `artifacts/checkpoints/legacy/early_runs/` |
| `res_4mln/` | Primi checkpoint A2C | 8,5 MiB | `artifacts/checkpoints/legacy/early_runs/` |
| `res_5mln/` | Primi checkpoint PPO | 13 MiB | `artifacts/checkpoints/legacy/early_runs/` |

Questi checkpoint non sono quelli selezionati dal runtime corrente. Vanno
comunque conservati come storia sperimentale.

### Grafici storici sciolti

- `distance_metrics.png`
- `distance_metrics1.png`
- `distance_metrics2.png`
- `distance_metrics3.png`
- `training_metrics.png`
- `training_metrics1.png`
- `training_metrics2.png`
- `sim_training_metrics.png`

Destinazione proposta:

```text
artifacts/reports/legacy/early_training_plots/
```

### Trace del profiler

La directory letterale `${data}/` contiene trace `NvStreamer-*.etli`, non un
dataset del progetto. Destinazione proposta:

```text
artifacts/profiling/legacy_nvstreamer/
```

Il nome anomalo deriva verosimilmente da una variabile shell non espansa.

## 5. Da verificare manualmente prima di qualsiasi movimento

| Elemento | Dimensione | Perche' richiede verifica |
|---|---:|---|
| `Data/` | 257 MiB | Contiene `4001.ckpt` e `real.ckpt`; bisogna confrontarli con i checkpoint TCC correnti |
| `video_raccolti/` | 9,9 GiB | Dataset episodico; potrebbe essere sorgente di training e non semplice output |
| `success_snapshots/` | 1,7 GiB | Grande raccolta di frame; serve stabilire quali run l'hanno prodotta e se e' ancora utile |
| `success_videos/` | 485 MiB | Video di successo; alcuni potrebbero essere materiale qualitativo utile |
| `install/` | 8 KiB | Output colcon con file tracciati; verificare se l'extension workflow lo richiede |
| `log/` | 4 KiB | Output colcon con file tracciato; probabilmente rigenerabile |
| `.venv/` | 16 MiB | Ambiente locale: puo' restare nella root ma deve rimanere ignorato |
| `.pytest_cache/` | 40 KiB | Cache rigenerabile; eliminabile dopo approvazione |

`success_snapshots/` e `success_videos/` sono chiaramente artifact, ma prima di
spostarli conviene creare un manifest con numero di file, range temporale e
run/checkpoint di provenienza, quando ricostruibile.

## 6. Ordine proposto delle operazioni

### Lotto A — rischio basso

1. Spostare i tre documenti tecnici in `docs/`.
2. Spostare il gruppo completo dell'evaluation del 13 novembre.
3. Spostare presentazioni generate, clip e grafici sciolti.
4. Spostare i 257 log in `artifacts/legacy_root/logs/` preservando i nomi.
5. Spostare le piccole directory SB3/TensorBoard gia' identificate.
6. Archiviare `res_*` e le trace NvStreamer.

Prima del lotto verra' generato un manifest TSV con percorso sorgente,
destinazione, dimensione e hash SHA-256 per ogni file mosso.

### Lotto B — dopo verifica manuale

1. Identificare i due checkpoint in `Data/` tramite hash e metadata.
2. Ricostruire la provenienza di `video_raccolti/`.
3. Inventariare `success_snapshots/` e `success_videos/`.
4. Decidere se rimuovere dal tracking gli output colcon `install/` e `log/`.

### Elementi esclusi da entrambi i lotti

- codice sotto `source/`;
- script e launcher sotto `scripts/`;
- test;
- configurazioni;
- checkpoint e dataset gia' collocati sotto `artifacts/`;
- qualsiasi elemento usato dal runtime corrente senza una verifica dei path.
