"""Helpers for bootstrapping Isaac Lab entrypoints before AppLauncher starts."""

from __future__ import annotations

import os


def pin_process_to_requested_cuda_device(device: str | None) -> str | None:
    """Pin the current process to the requested CUDA device.

    On some Isaac Sim / PyTorch setups, leaving CUDA visibility unpinned can make
    lazy CUDA initialization observe an invalid device index. When the caller asks
    for a single CUDA device and the process has not already been pinned through
    ``CUDA_VISIBLE_DEVICES``, expose only that physical GPU and remap the in-process
    device string to ``cuda:0``.
    """
    if device is None or "cuda" not in device:
        return device

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return device

    device_index = 0
    if ":" in device:
        suffix = device.split(":", 1)[1]
        if suffix:
            device_index = int(suffix)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
    return "cuda:0"


def get_best_available_checkpoint(experiment_dir: str) -> tuple[str | None, int | None]:
    """
    Rileva tutti i checkpoint attualmente presenti su disco e confronta le relative
    metriche su Tensorboard per caricare il checkpoint che ha performato meglio.
    """
    import glob
    # 1. Recupera i checkpoint presenti
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        return None, None
        
    ckpt_steps = []
    ckpt_map = {}
    for c in ckpts:
        base = os.path.basename(c)
        name, _ = os.path.splitext(base)
        try:
            step = int(name)
            ckpt_steps.append(step)
            ckpt_map[step] = c
        except ValueError:
            pass
            
    if not ckpt_steps:
        # Fallback all'ultimo in ordine alfabetico
        last_ckpt = sorted(ckpts)[-1]
        return last_ckpt, None

    # 2. Leggi eventi Tensorboard
    tb_dir = os.path.join(experiment_dir, "tb")
    event_files = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))
    if not event_files:
        # Nessun file Tensorboard, fallback all'ultimo checkpoint salvato
        latest_step = max(ckpt_steps)
        print(f"[XIRL Resolver] Nessun file Tensorboard trovato in {tb_dir}. Uso l'ultimo checkpoint su disco: {latest_step}.ckpt")
        return ckpt_map[latest_step], latest_step
        
    from tensorboard.backend.event_processing import event_accumulator
    
    # Raccoglie gli step valutati e i loro valori
    eval_data = [] # lista di tuple (step, value, metric_name)
    
    # Prova prima con Kendall's Tau in validazione
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(event_file, size_guidance={
            event_accumulator.SCALARS: 0,
        })
        try:
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            if "downstream/valid/kendalls_tau" in tags:
                for event in ea.Scalars("downstream/valid/kendalls_tau"):
                    eval_data.append((event.step, event.value, "Kendall's Tau"))
        except Exception:
            pass
            
    # Se non c'è Kendall's Tau, prova con Validation Loss (valore più basso è meglio)
    is_loss = False
    if not eval_data:
        is_loss = True
        for event_file in event_files:
            ea = event_accumulator.EventAccumulator(event_file, size_guidance={
                event_accumulator.SCALARS: 0,
            })
            try:
                ea.Reload()
                tags = ea.Tags().get("scalars", [])
                if "pretrain/valid/total_loss" in tags:
                    for event in ea.Scalars("pretrain/valid/total_loss"):
                        eval_data.append((event.step, event.value, "Validation Loss"))
            except Exception:
                pass
                
    if not eval_data:
        # Fallback all'ultimo checkpoint
        latest_step = max(ckpt_steps)
        print(f"[XIRL Resolver] Nessuna metrica valida trovata negli eventi Tensorboard. Uso l'ultimo checkpoint su disco: {latest_step}.ckpt")
        return ckpt_map[latest_step], latest_step

    # 3. Associa a ciascun checkpoint disponibile lo step valutato più vicino
    best_ckpt_step = None
    best_val = -1.0 if not is_loss else float("inf")
    best_metric_name = "Kendall's Tau" if not is_loss else "Validation Loss"
    best_eval_step = None
    
    for ckpt_step in ckpt_steps:
        # Trova la valutazione più vicina a questo checkpoint
        closest_eval = min(eval_data, key=lambda x: abs(x[0] - ckpt_step))
        eval_step, eval_val, metric_name = closest_eval
        
        if is_loss:
            if eval_val < best_val:
                best_val = eval_val
                best_ckpt_step = ckpt_step
                best_eval_step = eval_step
        else:
            if eval_val > best_val:
                best_val = eval_val
                best_ckpt_step = ckpt_step
                best_eval_step = eval_step
                
    print(f"[XIRL Resolver] Miglior checkpoint DISPONIBILE su disco: {best_ckpt_step}.ckpt")
    print(f"                (Valutato al passo {best_eval_step} con {best_metric_name}: {best_val:.6f})")
    
    return ckpt_map[best_ckpt_step], best_ckpt_step


def resolve_tcc_checkpoint(path_or_dir: str) -> str:
    """Resolves a TCC checkpoint path. If the input is a directory,
    or if we can locate an experiment directory, finds the best available checkpoint."""
    import glob
    if os.path.isfile(path_or_dir):
        return path_or_dir
        
    exp_dir = path_or_dir
    if not os.path.isdir(exp_dir):
        # Se l'utente ha passato un path che contiene checkpoints o termina con .ckpt
        if "checkpoints" in path_or_dir:
            exp_dir = path_or_dir.split("checkpoints")[0]
        else:
            exp_dir = os.path.dirname(path_or_dir)
            
    if os.path.isdir(exp_dir):
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            best_ckpt, best_step = get_best_available_checkpoint(exp_dir)
            if best_ckpt:
                return best_ckpt
            
            # Fallback al più recente se non ha trovato il best su tensorboard
            ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
            if ckpts:
                print(f"[XIRL Resolver] Fallback al checkpoint più recente su disco: {ckpts[-1]}")
                return ckpts[-1]
                
    return path_or_dir
