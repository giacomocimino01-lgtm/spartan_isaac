# Risultati, grafici e comandi di inferenza

Aggiornato al 9 ottobre 2026.

Tutti i path di questo documento si riferiscono al server universitario:

```text
/home/aiprah/Documents/m_dVrk
```

## Avvertenze per interpretare i risultati

- I grafici di training mostrano statistiche raccolte durante PPO, spesso con azioni stocastiche. Non vanno presentati come se fossero valutazioni deterministiche held-out.
- Le percentuali deterministiche riportate qui provengono dai file `best_deterministic.json`, dai ranking CSV o dai report di evaluation.
- Tutte le inferenze consigliate usano la maschera `minimal` e azioni deterministiche.
- Solo Phase 0 parte con gli anelli sul piano della board: per questa fase viene usata una randomizzazione di scala `0.75`.
- Phase 1–4 partono invece dalla condizione di successo della fase precedente; non si deve applicare direttamente a esse la randomizzazione degli anelli di Phase 0.
- Phase 4 raggiunge il 98% quando viene valutata dalla posa canonica standalone, ma non è ancora robusta alla posa articolare lasciata dal full task. Il full task 0–4 va quindi presentato come esperimento in corso, non come successo end-to-end definitivo.

## Grafici principali da mostrare

### Confronto uniforme Phase 0–2, tre seed

Questi sono i grafici più adatti per confrontare le fasi con lo stesso protocollo da 1M PPO e tre seed:

```text
/home/aiprah/Documents/m_dVrk/artifacts/reports/controlled_serial_minimal_1m_3seed/learning_curves_success.png
/home/aiprah/Documents/m_dVrk/artifacts/reports/controlled_serial_minimal_1m_3seed/learning_curves_bimanuality.png
/home/aiprah/Documents/m_dVrk/artifacts/reports/controlled_serial_minimal_1m_3seed/controlled_protocol_report.md
```

Il primo confronta le curve di successo; il secondo mostra l'evoluzione della bimanualità. È il confronto scientificamente più omogeneo disponibile.

### Phase 0

Training minimal-mask con randomizzazione iniziale `0.75`:

```text
/home/aiprah/Documents/m_dVrk/artifacts/reports/random_sb3_log_sim_phase_0_adaptive075_minimalmask_finetune_3m/training_analysis_report.png
/home/aiprah/Documents/m_dVrk/artifacts/reports/random_sb3_log_sim_phase_0_adaptive075_minimalmask_finetune_3m/report_summary.json
```

Il report termina con un success rate di training del 91% negli ultimi 100 episodi. Per ricostruire l'intera evoluzione dei training Phase 0:

```text
/home/aiprah/Documents/m_dVrk/artifacts/reports/random_sb3_log_sim_phase_0_full_lineage/training_analysis_report.png
/home/aiprah/Documents/m_dVrk/artifacts/reports/random_sb3_log_sim_phase_0_full_lineage/report_summary.json
```

Benchmark storico deterministico a scala `0.75`: 49/50 successi. Attenzione: appartiene alla precedente linea `adaptive_random`, quindi va distinto dal successivo fine-tuning minimal-mask.

```text
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/phase0_scale075_50ep/best_deterministic.json
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/phase0_scale075_50ep/checkpoint_ranking.csv
```

### Phase 1

Training con recovery replay:

```text
/home/aiprah/Documents/m_dVrk/artifacts/reports/random_sb3_log_sim_phase_1_native_minimal_recovery_mixed_replay_probe_300k/training_analysis_report.png
/home/aiprah/Documents/m_dVrk/artifacts/reports/random_sb3_log_sim_phase_1_native_minimal_recovery_mixed_replay_probe_300k/report_summary.json
```

Valutazione deterministica completa, da quattro anelli sul piolo verde a due rossi/due blu:

```text
/home/aiprah/Documents/m_dVrk/artifacts/reports/phase1_complete_deterministic_evaluation/phase1_complete_evaluation_report.png
/home/aiprah/Documents/m_dVrk/artifacts/reports/phase1_complete_deterministic_evaluation/report_summary.json
```

Il risultato è 49/50 nella valutazione primaria e 48/50 nella replica. Il report aggrega descrittivamente 97/100, ma specifica correttamente che le due prove non costituiscono 100 episodi completamente indipendenti.

### Phase 2

Training completo da `red2_blue2` verso quattro anelli al centro:

```text
/home/aiprah/Documents/m_dVrk/artifacts/reports/random_sb3_log_sim_phase_2_native_minimal_dense_replay_stratified50_probe_400k/training_analysis_report.png
/home/aiprah/Documents/m_dVrk/artifacts/reports/random_sb3_log_sim_phase_2_native_minimal_dense_replay_stratified50_probe_400k/report_summary.json
```

Successivo esperimento di recovery replay:

```text
/home/aiprah/Documents/m_dVrk/artifacts/reports/random_sb3_log_sim_phase_2_native_minimal_recovery_mixed_replay_probe_300k/training_analysis_report.png
/home/aiprah/Documents/m_dVrk/artifacts/reports/random_sb3_log_sim_phase_2_native_minimal_recovery_mixed_replay_probe_300k/report_summary.json
```

Il checkpoint selezionato per il task completo ottiene 39/50 successi deterministici:

```text
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/phase2_native_minimal_full_probe_50ep/best_deterministic.json
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/phase2_native_minimal_full_probe_50ep/checkpoint_ranking.csv
```

### Phase 3

Training bimanuale minimal-mask per distribuire un anello qualsiasi su ciascun piolo colorato:

```text
/home/aiprah/Documents/m_dVrk/artifacts/reports/random_sb3_log_sim_phase_3_bimanual_minimal_bc80_replay_probe_400k/training_analysis_report.png
/home/aiprah/Documents/m_dVrk/artifacts/reports/random_sb3_log_sim_phase_3_bimanual_minimal_bc80_replay_probe_400k/report_summary.json
```

Ranking deterministico: 48/50 per il checkpoint selezionato.

```text
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/phase3_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/best_deterministic.json
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/phase3_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/checkpoint_ranking.csv
```

### Phase 4

Training canonico bimanuale minimal-mask per mettere due anelli fuori a destra e due fuori a sinistra:

```text
/home/aiprah/Documents/m_dVrk/artifacts/reports/random_sb3_log_sim_phase_4_bimanual_minimal_bc80_replay_probe_400k/training_analysis_report.png
/home/aiprah/Documents/m_dVrk/artifacts/reports/random_sb3_log_sim_phase_4_bimanual_minimal_bc80_replay_probe_400k/report_summary.json
```

Ranking deterministico dalla posa canonica: 49/50.

```text
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/phase4_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/best_deterministic.json
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/phase4_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/checkpoint_ranking.csv
```

Il seguente è invece un risultato negativo ma scientificamente interessante: il curriculum canonica→parked si è fermato a scala `0.25`, con success rate finale di training del 52%, e non ha migliorato l'inferenza dalla posa parked esatta.

```text
/home/aiprah/Documents/m_dVrk/artifacts/reports/random_sb3_log_sim_phase_4_pose_curriculum_replay_800000/training_analysis_report.png
/home/aiprah/Documents/m_dVrk/artifacts/reports/random_sb3_log_sim_phase_4_pose_curriculum_replay_800000/report_summary.json
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/phase4_pose_curriculum_replay_800000/parked_rank_50ep/checkpoint_ranking.csv
```

### Router TCC

Analisi del selettore di encoder basato sul primo frame, limitata all'esperimento Phase 0–2:

```text
/home/aiprah/Documents/m_dVrk/artifacts/reports/phase_router_phase012_first_frame/phase_router_analysis_report.png
```

## Video già disponibili e utili

### Phase 0 randomizzata a scala 0.75

Video deterministico riuscito con maschera minimal e randomizzazione verificata dal log:

```text
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/phase0_adaptive075_minimalmask_inference/final_seed42_success.mp4
```

Metriche e log associati:

```text
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/phase0_adaptive075_minimalmask_inference/final_seed42.csv
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/phase0_adaptive075_minimalmask_inference/final_seed42.log
```

### Phase 1

Tre esempi deterministici riusciti con maschera minimal:

```text
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/minimal_mask_phase_inference_videos_20260927/phase_1/episode_000_success_phase_1.mp4
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/minimal_mask_phase_inference_videos_20260927/phase_1/episode_001_success_phase_1.mp4
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/minimal_mask_phase_inference_videos_20260927/phase_1/episode_002_success_phase_1.mp4
```

La seguente cartella contiene anche due successi e un fallimento; è utile se si vuole discutere la variabilità e lo stato intermedio `3-1`:

```text
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/phase1_complete_deterministic_videos/
```

### Phase 2

Due successi e un fallimento deterministici:

```text
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/minimal_mask_phase_inference_videos_20260927/phase_2/episode_001_success_phase_2.mp4
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/minimal_mask_phase_inference_videos_20260927/phase_2/episode_002_success_phase_2.mp4
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/minimal_mask_phase_inference_videos_20260927/phase_2/episode_000_fail_phase_2.mp4
```

### Full task: video diagnostici, non successi completi

Con randomizzazione Phase 0 a scala `0.75`; l'episodio si arresta in Phase 2:

```text
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/full_task_phase01234_tcc_home_phase4_random075_seed42_20261008/episode_000_fail.mp4
```

Con condizione iniziale fissa; arriva a Phase 4, ma Phase 4 fallisce nonostante l'homing:

```text
/home/aiprah/Documents/m_dVrk/artifacts/evaluations/full_task_phase01234_tcc_home_phase4_fixed_seed42_20261008/episode_000_fail.mp4
```

Non esistono ancora video deterministici curati nelle cartelle di evaluation per Phase 3 e Phase 4. I comandi seguenti li generano in cartelle dedicate.

## Preparazione del terminale

Eseguire questo blocco una volta dopo la connessione SSH:

```bash
cd /home/aiprah/Documents/m_dVrk
source /home/aiprah/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

export REPO=/home/aiprah/Documents/m_dVrk
export IsaacLab_PATH=/home/aiprah/Documents/IsaacLab
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
```

I comandi sono foreground, senza `nohup`. Ogni comando registra tre episodi deterministici; sostituire `--episodes 3` con il numero desiderato.

## Inferenza del task completo Phase 0→4

Questo script usa la maschera minimal, il router TCC, una Phase 0 randomizzata a scala `0.75`, parking tra le fasi e homing prima di Phase 4:

```bash
FULL_OUT="$REPO/artifacts/evaluations/full_task_phase01234_random075_$(date +%Y%m%d_%H%M%S)"
RANDOMIZE_RINGS=1 \
  "$REPO/scripts/run_full_task_phase01234_home_probe.sh" "$FULL_OUT"

find "$FULL_OUT" -maxdepth 1 -type f -name '*.mp4' -print
```

Stato attuale atteso: il comando è corretto e produce video, CSV e transizioni del router, ma il sistema non è ancora robusto end-to-end. Phase 4 standalone non generalizza ancora alla posa lasciata dalle fasi precedenti.

## Inferenza indipendente Phase 0

La randomizzazione `0.75` è esplicita nel comando:

```bash
OUT="$REPO/artifacts/evaluations/showcase_phase0_random075_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
(
  cd "$OUT" || exit 1
  "$IsaacLab_PATH/isaaclab.sh" -p "$REPO/scripts/eval_run.py" \
    --checkpoint "$REPO/artifacts/checkpoints/ppo/dvrk_ppo_finale_phase_0_adaptive075_minimalmask_finetune_3m.zip" \
    --tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase0_tcc/checkpoints/4001.ckpt \
    --dataset_path /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_0 \
    --task_phase phase_0 \
    --randomize_rings --ring_randomization_scale 0.75 \
    --episodes 3 --seed 42 \
    --metrics_output "$OUT/episodes.csv" \
    --action_masking --action_mask_mode minimal \
    --quiet_steps --headless --device cuda:0
)
```

## Inferenza indipendente Phase 1

Condizione iniziale: tutti e quattro gli anelli sul piolo verde. Goal: due sul rosso e due sul blu.

```bash
OUT="$REPO/artifacts/evaluations/showcase_phase1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
(
  cd "$OUT" || exit 1
  "$IsaacLab_PATH/isaaclab.sh" -p "$REPO/scripts/eval_run.py" \
    --checkpoint "$REPO/artifacts/checkpoints/ppo/dvrk_ppo_best_terminal_distance_phase_1_native_minimal_recovery_mixed_replay_probe_300k.zip" \
    --tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase1_tcc/checkpoints/31000.ckpt \
    --dataset_path /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_1 \
    --task_phase phase_1 --phase1_initial_state stack_green \
    --episodes 3 --seed 42 \
    --metrics_output "$OUT/episodes.csv" \
    --action_masking --action_mask_mode minimal \
    --quiet_steps --headless --device cuda:0
)
```

## Inferenza indipendente Phase 2

Condizione iniziale: due anelli sul rosso e due sul blu. Goal: quattro anelli al centro della board.

```bash
OUT="$REPO/artifacts/evaluations/showcase_phase2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
(
  cd "$OUT" || exit 1
  "$IsaacLab_PATH/isaaclab.sh" -p "$REPO/scripts/eval_run.py" \
    --checkpoint "$REPO/artifacts/evaluations/phase2_native_minimal_full_probe_50ep/phase2_native_minimal_full_best_50ep.zip" \
    --tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase2_tcc/checkpoints/4000.ckpt \
    --dataset_path /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_2 \
    --task_phase phase_2 --phase2_initial_state red2_blue2 \
    --episodes 3 --seed 42 \
    --metrics_output "$OUT/episodes.csv" \
    --action_masking --action_mask_mode minimal \
    --quiet_steps --headless --device cuda:0
)
```

## Inferenza indipendente Phase 3

Condizione iniziale: quattro anelli al centro. Goal: un anello qualsiasi su ciascun piolo colorato.

```bash
OUT="$REPO/artifacts/evaluations/showcase_phase3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
(
  cd "$OUT" || exit 1
  "$IsaacLab_PATH/isaaclab.sh" -p "$REPO/scripts/eval_run.py" \
    --checkpoint "$REPO/artifacts/evaluations/phase3_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/phase3_bimanual_minimal_best_50ep.zip" \
    --tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase3_tcc/checkpoints/2600.ckpt \
    --dataset_path /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_3 \
    --task_phase phase_3 \
    --episodes 3 --seed 42 \
    --metrics_output "$OUT/episodes.csv" \
    --action_masking --action_mask_mode minimal \
    --quiet_steps --headless --device cuda:0
)
```

## Inferenza indipendente Phase 4

Condizione iniziale: un anello qualsiasi su ciascun piolo colorato. Goal: due fuori a destra e due fuori a sinistra.

Il comando usa la posa canonica standalone, cioè la configurazione sulla quale il checkpoint raggiunge 49/50. Non aggiungere `--phase_entry_park_reset` per una demo positiva: dalla posa parked la policy corrente non è robusta.

```bash
OUT="$REPO/artifacts/evaluations/showcase_phase4_canonical_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
(
  cd "$OUT" || exit 1
  "$IsaacLab_PATH/isaaclab.sh" -p "$REPO/scripts/eval_run.py" \
    --checkpoint "$REPO/artifacts/evaluations/phase4_bimanual_minimal_80ep_pipeline/ppo_rank_50ep/phase4_bimanual_minimal_best_50ep.zip" \
    --tcc_checkpoint /home/aiprah/Documents/tmp/xirl/sim_pretrain_runs/random_sim_phase4_tcc/checkpoints/2600.ckpt \
    --dataset_path /mnt/data/aiprah/data/random_sim_dataset_tcc/train/phase_4 \
    --task_phase phase_4 \
    --episodes 3 --seed 42 \
    --metrics_output "$OUT/episodes.csv" \
    --action_masking --action_mask_mode minimal \
    --quiet_steps --headless --device cuda:0
)
```

## Dove vengono salvati i nuovi video

Ogni comando standalone salva nella propria cartella:

```text
episode_000_success_phase_X.mp4
episode_001_success_phase_X.mp4
episode_002_success_phase_X.mp4
episodes.csv
```

Se un episodio fallisce, il nome contiene `fail` invece di `success`.

Per elencare rapidamente gli ultimi video generati:

```bash
find "$REPO/artifacts/evaluations" -type f -name '*.mp4' \
  -printf '%T@ %p\n' | sort -nr | head -n 20
```

Per copiare un video dal server al computer locale, eseguire dal terminale del computer locale:

```bash
scp aiprah@157.27.31.172:/path/assoluto/del/video.mp4 .
```

