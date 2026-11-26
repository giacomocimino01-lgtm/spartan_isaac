Calibrated TCC phase router
===========================

This is the incremental router described in Sections 5--9 of
``tcc_orchestration_report.md``.  It is intentionally separate from
``smart_phase_router.py``: there is no bootstrap credal set, conformal gate, or
OOD detector in this baseline.

Runtime rule
------------

The active phase ``k`` is known.  Every frame is encoded by the active encoder
and by every candidate encoder ``j != k``.  The implementation evaluates

.. math::

   \log E_k(x) = -\frac{\|\phi_k(x)-g_k\|^2}{\tau_k^{end}},

.. math::

   \delta_j(x) =
   \frac{\|\phi_j(x)-s_j\|^2}{\tau_j^{start}},\qquad
   P(start_j\mid x)=\sigma(a_j\delta_j(x)+c_j),

.. math::

   \log T_{k\to j}(x)=
   \log E_k(x)+\log P(start_j\mid x)+\log\pi(j\mid k).

There is deliberately no preliminary ``E_k`` threshold.  All candidate
encoders run on every frame, following the implementation choice agreed with
the supervisors.

Let ``j*`` and ``j(2)`` be the best and second-best candidates.  One frame
supports a transition only when all three conditions hold:

.. math::

   \log T_{k\to j^*} > \theta_k,

.. math::

   \log T_{k\to j^*}-\log T_{k\to j^{(2)}} > m,

.. math::

   \log P(start_{j^*}\mid x)>\theta^{start}.

The same candidate must pass for ``n`` consecutive frames.  A failed condition
resets the counter; a different supported candidate restarts it from one.
Parking and homing are executed only after the statistical decision has been
confirmed.

Calibration
-----------

``scripts/calibrate_phase_router.py`` performs the offline procedure:

* embeds every calibration frame with every phase-specific TCC encoder;
* pools the first/last ``L`` endpoint frames into start/goal prototypes;
* estimates start/end distance scales with leave-one-trajectory-out medians;
* fits one class-balanced monotone logistic start detector per phase;
* normalizes annotated transition counts into a sparse prior, when supplied;
* reports detector diagnostics and static validation start-choice accuracy;
* replays reconstructed validation trajectories over a grid of
  ``theta``, ``m``, start floors, and consecutive-frame counts;
* reports timing hit rate, choice accuracy, misses, false switches, and delay.

Example skeleton::

   python scripts/calibrate_phase_router.py \
     --phase phase_0 CKPT_0 CALIB_0 VALID_0 \
     --phase phase_1 CKPT_1 CALIB_1 VALID_1 \
     --phase phase_2 CKPT_2 CALIB_2 VALID_2 \
     --guard_transition phase_0 phase_1 \
     --guard_transition phase_1 phase_2 \
     --transition_count phase_0 phase_1 20 \
     --transition_count phase_1 phase_2 20 \
     --endpoint_frames 5 \
     --guard_frames 10 \
     --output_dir artifacts/evaluations/phase_router_calibrated

For sequential replay reconstructed from phase folders, corresponding segments
must reuse the same video-directory name across phase folders.  If they do not,
the script still fits and validates the detector but reports that boundary grid
search is unavailable.

Sparse transition prior
-----------------------

For annotated counts ``N[k,j]``, calibration stores

.. math::

   \pi(j\mid k)=\frac{N_{kj}}{\sum_l N_{kl}}.

Only positive observed edges are serialized.  When a row exists for the
current phase, every omitted outgoing edge has probability zero and is removed
from the candidate set.  A row with one observed successor therefore assigns
that successor probability one.  This is task-order knowledge in the score,
not an episode-history mask.

With one admissible successor there is no finite runner-up, so the margin is
``+inf`` and the margin condition is automatically satisfied.  In that case
transition timing is governed only by the total-score threshold, the standalone
start floor, and temporal confirmation; ``theta_k`` must therefore be retuned
with the sparse prior enabled.

An existing v1 artifact can receive an independently estimated prior without
refitting prototypes or logistic detectors::

   python scripts/set_calibrated_router_priors.py \
     --input calibrated_router.pt \
     --output calibrated_router_sparse_prior.pt \
     --transition_count phase_0 phase_1 20 \
     --transition_count phase_1 phase_2 20

Artifacts without transition counts retain the uniform-prior behaviour for
backward compatibility.

Full-task evaluation
--------------------

Use the fitted artifact and the validation-selected parameters::

   python scripts/eval_full_task.py \
     ... phase and policy arguments ... \
     --router_mode calibrated \
     --calibrated_router_artifact ARTIFACT/calibrated_router.pt \
     --transition_threshold phase_0 THETA_0 \
     --transition_threshold phase_1 THETA_1 \
     --transition_margin M \
     --start_probability_threshold P_START \
     --transition_consecutive_steps N

All non-terminal phases need a transition threshold.  The final phase has no
candidate successor, so its episode completion remains controlled by the
existing symbolic or TCC end-only success mode.

The evaluator writes ``transition_score_trace.csv`` with the per-frame score
decomposition, each Boolean condition, confirmation streak, and decision.  The
expected ``--sequence`` scores the chosen phase.  Candidate restriction is
controlled only by the transition prior: with no prior every phase except the
active one remains a candidate; with a sparse row only its positive-probability
edges are evaluated.

Files
-----

* ``source/m_dVrk/m_dVrk/hrl/calibrated_phase_router.py``: fitting and runtime;
* ``scripts/calibrate_phase_router.py``: offline calibration and replay;
* ``scripts/set_calibrated_router_priors.py``: attach annotated sparse priors;
* ``scripts/eval_full_task.py``: closed-loop integration;
* ``tests/test_calibrated_phase_router.py``: numerical and temporal tests;
* ``source/m_dVrk/m_dVrk/hrl/phase_router.py``: unchanged raw baseline;
* ``source/m_dVrk/m_dVrk/hrl/smart_phase_router.py``: later credal experiment,
  not used by this baseline.

Validation status
-----------------

Unit tests verify the formulas, all three conditions, candidate exclusion,
debounce, counter reset, and artifact round-trip.  Threshold values are not
scientific results until calibrated on held-out trajectories and confirmed on
closed-loop policy rollouts.
