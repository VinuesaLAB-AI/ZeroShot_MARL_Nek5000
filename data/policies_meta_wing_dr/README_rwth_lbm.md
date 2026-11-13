# LBM-trained policies N1 and N1+F for the NACA0012 wing (RWTH)

Kai Lagemann and Christian Lagemann

Two wall-blowing policies for the NACA0012 wing at Re_c = 2e5, trained with TD3 in a JAX lattice-Boltzmann solver and
run without retraining in NekRS. They use the same observation and action as `eval_model_lc_omega1_SS_BASE_dr` and
replace it directly in the NACA0012 set-up of 2026-11-06.

| Policy | Region (x/c, suction side) | Actor file | Training |
|---|---|---|---|
| N1 | 0.23–0.87 | `rwth_lbm_N1/logs/eval_model_rwth_lbm_N1.zip` | TD3 in the LBM channel at Re_tau ≈ 206 |
| N1+F | < 0.48 | `rwth_lbm_F/logs/eval_model_rwth_lbm_F.zip` | N1 fine-tuned with TD3 on the front block of the LBM wing |
|      | > 0.48 | `rwth_lbm_N1/logs/eval_model_rwth_lbm_N1.zip` | as N1 |

Run configurations: `conf/naca0012_200k/NACA0012-DR-rwthN1-constant-utau.yml` and
`conf/naca0012_200k/NACA0012-DR-rwthN1F-constant-utau.yml`. They are copies of `NACA0012-DR-originTD3-constant-utau.yml`
with only the policy entries and the control regions changed. No wall node lies at x/c = 0.48, so the two regions do not
overlap. `compile_path` points to `envs/cases/naca0012_200k_constant_utau`, the case of that set-up.

## Actor

- Observation (u_t'/u_tau, v_n'/u_tau) at y+ = 15 with the spanwise mean removed, in the order of the Nek observation
  (`source_solvers: nek`), as for the HydroGym policy.
- Network 2 -> 8 -> 1 with **tanh** hidden layer and tanh output, i.e. `policy_kwargs = {net_arch: {pi: [8]},
  activation_fn: Tanh}`. The HydroGym policy has the same size with ReLU.
- Action a in [-1, 1], wall-normal velocity a times the amplitude of the case, positive = blowing, zero net mass flux
  per spanwise line, new action every 31 solver steps (dt = 4e-5).
- The checkpoints were written with Stable-Baselines3 1.7.0, gym 0.24.1, torch 2.4.1 and Python 3.8 (the versions in
  `system_info.txt` of the HydroGym checkpoints). The critic is an untrained placeholder and is not used.

| File | sha256 | Source checkpoint (JAX, sha256) |
|---|---|---|
| `eval_model_rwth_lbm_N1.zip` | `785ae5577d88bfed6976d2ddf2423d2a4da092652e1ef4612c8d6a32407ea2eb` | N1 ep. 12, `cfe369d08e23df06…` |
| `eval_model_rwth_lbm_F.zip` | `f6d83fcbd981ad971fa1d31feba0cf57bd9065be55c693eacf06f1ce647c5868` | F ep. 8, `a5332648a607584c…` |

## Embedded (Fortran) evaluation: change in `src/lib/pol_export.py`

`extract_actor` wrote ReLU for every hidden layer of a TD3/DDPG/SAC actor, independent of `activation_fn`. For these
actors the `.pol` file then evaluated a different network, and the embedded run gives no warning because the reference
table (`.chk`) is only used by `utils/pol_selftest.sh`. The hidden-layer activation is now read from `policy_kwargs`
in the checkpoint (ReLU when absent, an error for anything other than ReLU or tanh). The export of the existing ReLU
checkpoints is unchanged: the `.pol` of `eval_model_lc_omega1_SS_BASE_dr` is identical to the previous one, apart from
the time stamp, and so are its weight bits.

`utils/pol_selftest.sh` (`sb3_to_f77.py --utau 1 --amp 1`, 20 000 reference rows, `pol_core.f` built with gfortran),
maximum |a_Fortran − a_SB3|:

| Checkpoint | before the change | after the change |
|---|---|---|
| `eval_model_lc_omega1_SS_BASE_dr` | 6.0e-7 | 6.0e-7 |
| `eval_model_rwth_lbm_N1` | 1.46 | 2.4e-7 |
| `eval_model_rwth_lbm_F` | 1.44 | 2.4e-7 |

## Checks

1. Loaded as in `MetaPolicy._load_policy` (SB3 1.7.0, observation space (2, 1, 1)), the checkpoints reproduce the
   reference actions of the original JAX actors within 2.1e-7 (64 test states each).
2. The embedded export reproduces SB3 `predict` within 2.4e-7 (table above).
3. The two zip files of this directory ran in NekRS 26.0, set-up as below, with 69 595 agents. They were loaded with
   `TD3.load` and evaluated with `predict(deterministic=True)`. Each run covered 100 decisions from the controlled state
   at t = 30.48 c/U and was compared with the same restart driven by the original actors. Relative difference of the
   drag force, mean over the 100 decisions: 2.8e-7 (N1) and 7.6e-7 (N1+F); largest single decision: 1.1e-5 and 5.2e-5.
   Actuator power: 7.8e-7 and 7.7e-7. This check covers the Python deployment. The Fortran (embedded) path is covered
   by check 2. We did not run Nek5000 itself.

## Results in NekRS (HydroGym deployment)

NekRS 26.0 with the layout of the NACA0012 set-up of 2026-11-06 (X2Utau of `drl_init.f`, amplitude 0.0624334733,
31 steps per action at dt = 4e-5). Control starts at t = 28 c/U; means over t_c = 1.0–3.5 c/U, paired with the
uncontrolled run. Wing values in the metric of the roll-outs: friction integrated over the statistics stations
0.0197 <= x/c <= 0.995 and pressure over the full surface, normalised by C_Df0 = 0.01049 and C_D0 = 0.01184.

| Law | strip DR_30 (%) | wing ΔC_Df (%) | wing ΔC_D (%) | δC_Dp (1e-5) |
|---|---|---|---|---|
| HydroGym TD3 (`eval_model_lc_omega1_SS_BASE_dr`) | 37.8 | 10.5 | 7.3 | +23.6 |
| N1 | 32.9 | 8.9 | 8.0 | −1.1 |
| N1+F | 35.5 | 9.8 | 8.7 | +0.8 |
| opposition control, y+ = 15 | 28.7 | 7.9 | 6.9 | +0.8 |

For the HydroGym policy and opposition control, these NekRS values agree with the Nek5000 drag records within 0.5
points.
