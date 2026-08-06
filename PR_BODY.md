Centralize artifacts and make scripts config-driven

Summary
- This branch centralizes experiment artifacts under a single `artifacts/` tree and introduces a small, optional `configs/defaults.yaml` to configure `artifacts_dir` and other paths.
- Scripts and evaluation utilities now prefer the configured `ARTIFACTS_DIR` while preserving backward-compatible fallbacks.

Key changes
- `scripts/*`, `evaluation/*`: prefer `artifacts/` for logs, checkpoints, datasets, and reports.
- `source/m_dVrk/m_dVrk/hrl/`: centralized constants, wrapper, tcc helper, and reward/success logic.
- `configs/defaults.yaml`: default paths and small experiment settings.
- `.github/workflows/ci.yml`: run pytest + ruff on PRs.

Testing performed
- Local smoke tests: `PYTHONPATH=source pytest -q -k 'not isaacsim_ci'` → 2 tests passed.
- Performed a repository scan for legacy hard-coded paths and updated evaluation and scripts where found.

How to review
- Check `CHANGELOG.md` for a summary of intent.
- Run the smoke tests locally in the IsaacLab environment.
- Verify that artifacts are saved under `artifacts/` when `configs/defaults.yaml` sets `artifacts_dir`.

Notes
- The branch contains many refactors; if you prefer smaller PRs, I can split changes into multiple focused PRs (e.g., core HRL refactor, scripts updates, CI).
