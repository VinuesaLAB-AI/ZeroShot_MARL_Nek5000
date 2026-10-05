# NEK5000 multi-agent reinforcement learning for power saving

Train and evaluate flow controllers with Python/Stable-Baselines3 coupled to
NEK5000, or evaluate an exported policy directly inside Nek. The repository
contains channel and wing cases, classical controllers, reward monitoring, and
tools for processing solver time series and recorded policy trajectories.

## Setup

Run the commands below from the repository root:

```bash
cd /home/yuninw/codes/drl/1_Nek/nek_power_saving
```

Compilation requires Bash, GNU Make, a Fortran compiler, and MPI C/Fortran
wrappers (`mpicc`, `mpif77`). Use the same MPI installation for the solver and
`mpi4py`. The launchers source `~/.bashrc.openmpi_ucx` and
`~/.bashrc.miniforge`; adapt these profiles to your machine.

For a new installation, these helpers install software and write shell
profiles. Inspect their installation paths before running them:

```bash
./utils/install_mpi.sh > log.mpi 2>&1
./utils/install_miniforge.sh > log.miniforge 2>&1
./utils/install_auto_env.sh nek > log.python 2>&1
```

The installer creates a Python 3.8 environment with the project's SB3 1.x,
Gym 0.24, PettingZoo 1.19, and SuperSuit 3.5 stack. See
[requirements.txt](requirements.txt) and
[install_auto_env.sh](utils/install_auto_env.sh) for dependency ranges and
installer pins. To register this checkout in an existing environment:

```bash
source ~/.bashrc.openmpi_ucx
source ~/.bashrc.miniforge
python -m pip install -e . --no-deps
python utils/patch_supersuit.py
```

The SuperSuit patch reuses the environment when `num_envs == 1`, avoiding a
copy of the MPI-connected environment. Restore it with
`python utils/patch_supersuit.py restore`.

Extract the solver distributions once, before the first compilation:

```bash
./utils/initialize_solver.sh
```

This unpacks the four archives under `envs/solver/`. Meshes, restart fields,
and trained checkpoints must also be available at the paths in your YAML.
Large shared meshes can be supplied through `simulation.shared_data_path`;
`shared_data_link: true` stages those files as symlinks.

## Compile a case

**Pass the case directory with `--path`, and choose `--solver drl` or
`--solver raw`.** Build the directory named by `simulation.compile_path` in
the configuration you intend to run.

```bash
source ~/.bashrc.openmpi_ucx

# Coupled training/evaluation: produces envs/cases/mini_channel/nek5000
./utils/compile_case.sh --path envs/cases/mini_channel --solver drl

# Embedded evaluation: produces envs/cases/mini_channel/nek5000_solo
./utils/compile_case.sh --path envs/cases/mini_channel --solver raw

# Large-channel examples
./utils/compile_case.sh --path envs/cases/lc_n7 --solver drl
./utils/compile_case.sh --path envs/cases/lc_n7 --solver raw
./utils/compile_case.sh --path envs/cases/lc_n7_onlyU --solver raw
./utils/compile_case.sh --path envs/cases/lc_n7_onlyV --solver raw
./utils/compile_case.sh --path envs/cases/lc_n7_UcombV --solver raw

# Wing examples; the case script selects the v17 solver and naca_wing name
./utils/compile_case.sh --path envs/cases/naca4412_ref_PSSS_2586 --solver drl
./utils/compile_case.sh --path envs/cases/naca4412_ref_SS_2540_vonly --solver raw

# Experimental Hasegawa/normalised-pressure reward case
./utils/compile_case.sh --path envs/exp_cases/lc_n7_jfm_normp --solver raw
```

| Case family | `--solver drl` → `nek5000` | `--solver raw` → `nek5000_solo` |
| --- | --- | --- |
| Channel (`v19`) | `KTH_DRL_Framework` | `KTH_Framework` |
| Wing (`v17`) | `DEC_DRL_Nek5000` | `DEC_Nek5000` |

The wrapper cleans and rebuilds the selected executable, answers `N` to
third-party cleanup prompts, and preserves the other executable. It verifies
that cleaning removed the old binary and compilation produced an executable
before reporting success. Legacy case scripts can mask compiler errors, so use
this wrapper for full rebuilds and inspect the compiler output on failure.
The wrapper also protects `nek5000` from the v17 build system's hard-coded
removal when building `nek5000_solo`.

`--solver drl` is the default. With no arguments the wrapper builds
`envs/cases/mini_channel`. `--m` remains an alias for `--path` and takes a
directory path. The old `--version` and `--case_name_v17` options are no longer
accepted: each case's `compile_script` already contains those settings.
`simulation.solver_version: v17` is still required in a wing **run config**.

For an incremental build inside an already configured case, use
`bash ./compile_script --solver drl --all` (or `raw`). Use the wrapper when
switching backends or when both executables must be preserved. Changes to
`SIZE`, observation dimensions, or preprocessor flags require recompilation.
Keep `SIZE`/`inc_src/DRL` consistent with the YAML's `lx1`, `TOTCTRL`,
`npl_state`, mesh dimensions, and MPI capacity.

## Coupled training and evaluation

The coupled executable uses `simulation.nproc` Nek ranks plus one Python
controller rank. Launchers perform initialization and construct the MPI command.
`--config` takes a YAML path relative to the repository root or an absolute path.
Use repository-root-relative paths for fields inside the YAML, including
`compile_path`, `restart_folder`, and `policy_file`; the coupled restart path
does not support an absolute `restart_folder`.

```bash
# Train from scratch; omit --load-agent to use the YAML value
./execs/channel-run.sh --config conf/mini_channel/MC-ng-111.yml \
    --mode train --load-agent False

# Resume training using the checkpoint selected by runner.policy
./execs/channel-run.sh --config conf/mini_channel/MC-ng-111.yml \
    --mode train --load-agent True

# Evaluate environments 1 and 2
./execs/channel-run.sh --config conf/mini_channel/MC-ng-111.yml \
    --mode evaluate --nenv 2 --iostep 10000 --smpstep 12

# Continue a sweep with environments 5 through 8, inclusive
./execs/channel-run.sh --config conf/mini_channel/MC-ng-111.yml \
    --mode evaluate --env-start 5 --nenv 8

# Multi-region wing evaluation
./execs/wing-run.sh --config conf/naca4412_200k/NACA4412-SHAP-Vel-2540.yml
```

`runner.policy` is the **full checkpoint filename stem**, such as `best_model`
or `mc_nes_nek-rl_model_500_steps`; no run-name prefix is added. Channel
checkpoints are resolved under `runs/<agent_run_name>/logs/`. Wing meta configs
use lists in `runner` for regions, checkpoint names, algorithms, scaling, and
update intervals.

Use `--dry-run` to inspect commands without launching MPI, and `--help` for
all flags. Repeated/comma-separated `--config` values run a sweep. For example,
`--extra "runner.seed=7 runner.nb_episodes=10"` supplies configuration overrides.
`--nenv` is the last environment index, not the number of parallel environments.

## Nek-only evaluation (embedded policy)

`nek-solo-run.sh` prepares the run in Python, exports `actor_*.pol`, and writes
`drl_policy.in`. The simulation then runs entirely in Nek with exactly `nproc`
MPI ranks. It requires the **raw** build (`nek5000_solo`). Initialization sets
`UPARAM(10)=1` for channels or `PARAM(91)=1` for wings.

```bash
# Learned large-channel controller; requires its checkpoint and restart fields
./execs/nek-solo-run.sh --config conf/omega1_BASE/solo/nes.yml

# Classical controllers; no SB3 checkpoint is needed
./execs/nek-solo-run.sh --config conf/omega1_BASE/solo/base.yml
./execs/nek-solo-run.sh --config conf/omega1_BASE/solo/oc.yml
./execs/nek-solo-run.sh --config conf/omega1_BASE/solo/oc_u.yml
./execs/nek-solo-run.sh --config conf/omega1_BASE/solo/oc_uvcomb.yml

# Continue from the newest complete local solver checkpoint
./execs/nek-solo-run.sh --config conf/omega1_BASE/solo/nes.yml --resume
```

The analytic laws are `BL` (constant blowing/suction; zero amplitude gives a
baseline), `OC` (oppose wall-normal velocity), `OC_U` (`-alpha*u'`), and
`OC_UVCOMB` (`-alpha*(0.5*u' - v')`). Scaling follows the configuration; the
provided `oc_u.yml` and `oc_uvcomb.yml` use unnormalised observations.

| Case | Policy input | Example configuration |
| --- | --- | --- |
| `lc_n7` | `[u', v']`, `npl_state: 2` | `conf/omega1_BASE/solo/nes.yml` |
| `lc_n7_onlyU` | `[u']`, `npl_state: 1` | `conf/omega1_BASE/solo/oc_u.yml` |
| `lc_n7_onlyV` | `[v']`, `npl_state: 1` | `conf/omega1_BASE/exp_vonlys/lc-omega1-SS-nes-org-vonly.yml` |
| `lc_n7_UcombV` | `[u', v']`, `npl_state: 2` | `conf/omega1_BASE/solo/oc_uvcomb.yml` |
| `naca4412_ref_SS_2540_vonly` | one wall-normal component | adapt a v17 meta config and one-input checkpoint |

In `lc_n7_onlyV`, the recorder additionally stores both `[u', v']` for analysis;
the controller and MPI observation remain one-dimensional. The U-only and
wing V-only recorders currently store one observation component.

Useful embedded options:

- `--nb-interactions N`, `--rec-freq N`, and `--precision 4|8` override run length,
  recorder cadence, and network accumulation precision (default: 8).
- `--mv-data yes` archives the previous environment **before** preparing its
  replacement, including fields, recorder segments, and policy provenance.
- `--skip-prepare` reuses a prepared run and its exported policies. It cannot be
  combined with `--resume`, which needs preparation to select a checkpoint.
- `embedded.coupled_recorder: true` also enables `drlrec/` output in coupled
  evaluation. It is off by default.

Embedded export supports the standard ReLU-hidden TD3/DDPG actors and
deterministic SAC actors. PPO remains a coupled-mode option. The new
`conf/zero_layer_ddpg_policy.yml` selects an actor with no hidden layers;
`conf/1Layer_1Node_ddpg_policy.yml` selects one hidden layer with one neuron.
**Do not export policies trained with `conf/default_ddpg_linear_policy.yml`
yet:** the exporter currently substitutes ReLU for their Identity hidden
activation, changing the controller. Use those policies through coupled Python
evaluation.

See [embedded_policy.md](temp/docs/embedded_policy.md) for formats, scaling,
region scheduling, recorder semantics, and resume behavior.

## Rewards

Choose `runner.reward_fn` in the YAML:

| Mode | Reward |
| --- | --- |
| `dudy` | `1 - mean(dU/dy) / dUdy_ref` |
| `net_gain` | `alpha*R_tau + beta*R_pw + gamma*R_v3` |

The net-gain components measure drag reduction and the pressure-work and
kinetic-energy costs. Raw values and averaging are computed by the case's
Fortran reward implementation; weights are applied when forming the reward.
Experimental `jfm`/normalised-pressure cases use their own reward variants.

```yaml
runner:
  reward_fn: net_gain
  reward_alpha: 1.0
  reward_beta: 1.0
  reward_gamma: 1.0
```

**Channel cases:** both reward paths are compiled. Initialization writes
`userParam09` (`0=dudy`, `1=net_gain`), so switching reward modes requires no
recompilation. `GAINMONITOR` remains a build flag for the extra reward monitor.

**Wing cases:** reward selection still uses the `NETGAIN` flag in the case's
`compile_script`. For coupled evaluation it must match the Python reward
mode: one reward buffer for `dudy`, three for `net_gain`. A mismatch can hang
the MPI exchange. Rebuild after changing the flag.

Channel **evaluation** defaults to `net_gain` with all three weights set to
`1.0`, overriding the YAML on both initialization and evaluation. Set
`--reward-fn` and `--alpha/--beta/--gamma` to change that. Training uses the YAML
values; the solo launcher also uses the configured reward values.

Details: [runtime reward switch](temp/docs/readme_runtime_reward_switch.md) and
[net-gain derivation](temp/docs/readme_POL_netgain.md). The older derivation's
compile-time instructions apply to wings; use the runtime switch for channels.

## Submit to SLURM

```bash
# Generate a job; append --submit to submit it
./execs/sjob-gen.sh --case channel --config conf/mini_channel/MC-ng-111.yml \
    --mode train -J ng-111 -p batch -t 24:00:00 --begin +2h

# Nek-only job; payload options follow the bare --
./execs/sjob-solo --config conf/omega1_BASE/solo/oc_u.yml \
    -J oc-u -t 24:00:00 -- --nb-interactions 20000
```

Generated jobs go to `execs/sjobs/<job-name>.sh`. `--print` prints the script
without writing it. Node sizing defaults to `auto`: it uses `nproc + 1` ranks
for coupled jobs and `nproc` for solo jobs. All payload flags can be forwarded
after `--`. See [execs/readme.md](execs/readme.md) for the full job interface.

## Monitoring and post-processing

```bash
python utils/monitor_runs.py
python utils/read-history.py --id 20011 --mean
tensorboard --logdir runs/20011/train/history/tensorboard

# Export reward-component learning curves as NPZ and MATLAB files
python post_processing/export_reward_components.py \
    --case 20011=net-gain --format both

# Stitch solver time series and save spectra in the results tree
python post_processing/tsrs_stitch.py --case_id 20011 --runs-dir runs
python post_processing/tsrs_spectra.py 20011 --runs-dir runs \
    --outdir post_processing/Figs --results

# Post-hoc GLL zero-mean action analysis of an embedded evaluation
python post_processing/znmf_actuation.py lc_omega1_SS_nes_org \
    --support active --policy-map-u-range -6 6 --policy-map-v-range -1 1

# Gather derived data, config, and evaluated model provenance
./utils/collect-results --run_name lc_omega1_SS_nes_org
```

`simulation.y_planes` selects diagnostic TSRS planes independently of the DRL
`y_sensing` plane. More planes require sufficient `lhis` in `SIZE`; rebuild
after increasing it. Multi-plane spectra, space/time correlations, convection
velocity, snapshots, and wall-shear/velocity correlations are available through
[README_tsrs.md](post_processing/README_tsrs.md) and
[tsrs_cases.ipynb](post_processing/tsrs_cases.ipynb). Wall-shear reconstruction
requires the appropriate near-wall planes; missing planes are reported.

`znmf_actuation.py` exports raw and GLL-weighted zero-mean actions, conditional
action maps, observation PDFs, sensitivity curves, and direct policy surfaces
from deployed `actor_*.pol` files. Its correction is for analysis: recorder
actions and direct policy surfaces are **pre-ZNMF**, and this weighted mean is
not an exact reconstruction of the solver's applied boundary field. On a
one-input policy use `--policy-map-one-input u|v` to select the plotted input
explicitly. The full CLI's trajectory maps currently require **two recorded
observation components**, even though its direct policy-surface function
supports one-input actors; U-only/wing V-only records cannot use the full CLI.

The DRL recorder notebooks and `postlib/drlrec.py` also save lightweight,
plot-ready MATLAB products. See
[drlrec_processed_products.md](temp/docs/drlrec_processed_products.md).

Archive raw evaluation output with:

```bash
./utils/mv-data --case_name tcf --run_name lc_omega1_SS_nes_org --id 001
```

## Files and run layout

```text
conf/                         run and policy configurations
envs/cases/                   active channel and wing case sources
envs/exp_cases/               experimental channel/reward variants
envs/solver/                  four Nek solver trees and their archives
src/                          nek_MARL, meta_MARL, environments, shared utilities
execs/                        launchers and SLURM job generation
utils/                        setup, compilation, export, monitoring, archiving
post_processing/              trajectory, reward, time-series and spectra tools
temp/docs/                    detailed implementation/workflow references
log-files/                    launcher and SLURM logs
.caches/RUN_PATH_<run>.txt     initialization-to-launcher path handoff
runs/<run>/
  logs/                       saved SB3 checkpoints
  train/history/              live reward logs and TensorBoard data
  history/current_history/    most recently preserved training history
  history/roundXXX/            older training histories
  eval/env_XXX/               solver output, exported policies and drlrec/
  eval/history/               coupled evaluation reward logs
data/results/<case>/<run>/    archived output and derived results with provenance
```

Starting training preserves the previous history and rebuilds `train/`.
Post-processing reads the persistent histories and the live `train/history`.
Completed evaluations retain their evaluated model or exported policy files
alongside their data; `collect-results` gathers these into the results tree.

## Configuration migration and validation

Some older YAMLs still point to experimental cases under `envs/cases/`, although
the sources now live in `envs/exp_cases/`. This affects `mini_channel_v`,
`mini_channel_uv_invest`, and several `lc_*_jfm*` experiments. Update
`simulation.compile_path` to the actual directory before running them. In
particular, `conf/mini_channel/MC-nes.yml` needs
`envs/exp_cases/mini_channel_uv_invest`. Experimental cases do not all support
embedded mode; check their `compile_script` and policy sources before selecting
`--solver raw`. Many large-channel/wing restart datasets are external to this
checkout and must be supplied separately.

For offline policy validation against a local checkpoint:

```bash
./utils/pol_selftest.sh --ckpt runs/mc_nes_nek/logs/best_model.zip \
    --config conf/mini_channel/MC-nes.yml
python post_processing/test_explore.py
python -m pytest post_processing/test_znmf_actuation.py
python utils/test_compile_case.py
```

The self-test uses `gfortran` without Nek/MPI and compares 20,000 observations
against SB3. It reads scaling settings from the YAML, so the old compile path
in that particular config does not affect this offline test. The ZNMF test
command requires `pytest`; the history and compile-wrapper tests also run
directly without it.

### Validation snapshot: 2026-10-05

Checks were performed in an isolated copy, using the installed MPI/Python
environment and existing third-party solver libraries:

| Check | Result |
| --- | --- |
| All 18 active cases, both `drl` and `raw`, plus `lc_n7_jfm_normp` in both modes | 38 builds passed |
| Backend switching after the wrapper fix | Channel and v17 wing binaries preserved byte-for-byte |
| Minimal-channel MPI evaluation | Four interactions each: coupled `dudy`, coupled `net_gain`, and embedded TD3 passed |
| Embedded actor vs SB3 | 20,000 samples; maximum float64-accumulation action difference `8.759e-8` |
| Analytic `BL`, `OC`, `OC_U`, and `OC_UVCOMB` actors | Fortran replay passed for 200 samples per law; maximum error below `5e-10` |
| History, ZNMF/policy-surface, and compile-wrapper regressions | 8 + 4 + 7 checks passed |
| ZNMF CLI on the embedded run's recorder output | MATLAB export and all four figures generated |
| Launcher dry runs and SLURM `--print` output | Commands generated; printed scripts pass `bash -n`; no jobs submitted |
| Configuration schemas | All 98 local simulation YAMLs merged successfully with their schemas |

The audit found **24 configurations with stale `compile_path` values** and
**86 configurations whose restart folders are absent locally**. Schema
validation alone does not establish that a run has its inputs. The path
migration above, the Identity-activation export limitation, and the
two-component requirement for trajectory action maps remain applicable.
Absolute restart paths also fail in coupled evaluation; use root-relative paths.

This check fixed compile-wrapper failure reporting/default paths and v17 binary
preservation, Python 3.8 compatibility in policy-surface evaluation, and the
SLURM generator's status message leaking into `--print` script output. Long
training runs and production wing/large-channel simulations were not run;
their numerical behavior is not established by these compile and smoke checks.
