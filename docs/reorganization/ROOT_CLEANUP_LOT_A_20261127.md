# Root cleanup — lotto A

Data: 27 novembre 2026.

Branch: `refactor/repository-layout-20261127`, derivato dallo snapshot di
sicurezza `4d34c7d`.

## Scopo

Rimuovere dalla root esclusivamente documenti, log e output sperimentali gia'
classificati, senza cambiare codice, configurazioni runtime, checkpoint
selezionati o dataset da verificare.

## Contenuto

- 566 file inventariati, per 1.061.365.075 byte.
- Manifest preventivo con sorgente, destinazione, dimensione e SHA-256:
  `manifests/root_cleanup_lotA_manifest_20261127.tsv`.
- Documenti tecnici spostati sotto `docs/`.
- Evaluation full-task sciolta del 13 novembre ricomposta sotto
  `artifacts/evaluations/legacy_root/`.
- Log, media, presentazioni, profiler trace e primi checkpoint spostati sotto
  sottodirectory descrittive di `artifacts/`.
- `nohup.out` rimosso dal tracking Git ma conservato negli artifact.

## Esclusioni intenzionali

Non sono stati spostati:

- `Data/`;
- `video_raccolti/`;
- `success_snapshots/`;
- `success_videos/`;
- `install/` e `log/`;
- `.venv/` e `.pytest_cache/`;
- il codice in `source/`, `scripts/`, `evaluation/`, `tests/` e `configs/`;
- gli artifact che erano gia' organizzati prima del lotto.

## Regola Git

L'intero contenuto di `artifacts/` resta locale e ignorato, salvo
`artifacts/.gitkeep`. I documenti Markdown sotto `docs/` e tutti i test Python
sono invece versionabili.

## Verifica finale

- Tutte le 566 destinazioni sono state verificate contro dimensione e SHA-256
  registrati prima dello spostamento.
- Tutte le 566 sorgenti originarie risultano assenti dopo lo spostamento.
- Le directory escluse intenzionalmente risultano ancora presenti.
- `git diff --cached --check` non segnala errori di formattazione.
- Suite: 99 test superati con `env_isaaclab`.
