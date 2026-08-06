# Changelog — feature/artifacts-centralize

## Unreleased

- Centralize artifact and config paths under `artifacts/` and `configs/defaults.yaml`.
- Updated training/eval/collection scripts to prefer `ARTIFACTS_DIR` (config-driven).
- Added `configs/defaults.yaml` and `configs/npz_metadata_schema.json` (NPZ metadata validation schema).
- Centralized HRL constants and wrapper code under `source/m_dVrk/m_dVrk/hrl/`.
- Consolidated controllers under `source/m_dVrk/m_dVrk/controllers/` and fixed state-machine issues.
- Added CI workflow `.github/workflows/ci.yml` to run tests and linting on PRs.
- Removed legacy symlinks and migrated outputs to `artifacts/` with backward-compatible fallbacks.

### Notes for reviewers

- Focus review on `scripts/*` and `evaluation/*` changes to confirm path handling and fallbacks.
- Verify `configs/defaults.yaml` values match your expected artifact layout.
