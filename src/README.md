# Source code

See the [project README](../readme.md) for setup, compilation, and runnable
examples. Launch from the repository root through `execs/channel-run.sh`,
`execs/wing-run.sh`, or `execs/nek-solo-run.sh`.

| Module | Purpose |
| --- | --- |
| `configs.py`, `configs_meta.py` | Channel and wing configuration schemas, including reward and embedded-policy settings |
| `nek_MARL/`, `meta_MARL/` | CLI entry points (`nek-marl`, `meta-marl`, or `python -m ...`) |
| `initial.py`, `initial_meta.py` | Prepare run directories and preserve previous history |
| `run.py`, `evaluate.py`, `eval_meta.py` | Training and deterministic channel/wing evaluation |
| `nek_marl.py`, `nek_marl_meta*.py` | PettingZoo environments and MPI communication |
| `MetaPolicy.py` | Per-region wing policies and update scheduling |
| `lib/nek_utils.py` | Stage solver inputs, select runtime modes, manage restarts, and export embedded policies |
| `lib/pol_export.py` | Write learned/analytic `.pol` actors and `drl_policy.in` |
| `lib/AFC.py` | Classical `OC`, `OC_U`, `OC_UVCOMB`, `BL`, and sinusoidal controllers |
| `lib/sb3_utils.py` | Model initialization, callbacks, MPI splitting, and evaluated-checkpoint archiving |
| `lib/reward_logger.py` | CSV reward/component logging |
| `lib/writer_int_pos.py` | Diagnostic sampling points, including multiple channel `y_planes` |

Channel reward selection uses runtime `UPARAM(9)`; wing cases still select the
reward with the `NETGAIN` build flag. Embedded evaluation needs a raw solver
build and uses `UPARAM(10)` (v19) or `PARAM(91)` (v17).

The policy exporter assumes ReLU hidden activations for learned TD3/DDPG/SAC
actors; Identity-hidden policies and PPO should use coupled evaluation. See
[embedded policy details](../temp/docs/embedded_policy.md).
