# GEAR-SONIC-Kengo

An unofficial research adaptation of NVIDIA
[GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl)
and [GEAR-SONIC](https://nvlabs.github.io/GEAR-SONIC/) for the 23-DoF
Galaxea Kengo embodiment.

> 中文：本仓库提供 Kengo 的动作转换与 SONIC 过滤、tracker/controller
> 训练、最佳 checkpoint 选择、单体 ONNX 导出和 MuJoCo sim2sim 链路。
> 机器人资产、动作数据、模型和测试视频不随公开代码分发；目前也不包含
> 可直接驱动 Kengo 真机的传输与安全层。

[![Repository](https://img.shields.io/badge/repository-public-brightgreen.svg)](https://github.com/I3NLi/GEAR-SONIC-Kengo)
[![Robot](https://img.shields.io/badge/Kengo-23--DoF-blue.svg)](#implemented-scope)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.3.x-orange.svg)](https://github.com/isaac-sim/IsaacLab)
[![License](https://img.shields.io/badge/source-Apache--2.0-76B900.svg)](LICENSE)

## Results first

The latest locally validated policy is the private step-25,450 checkpoint
(training `Mean rewards = 20.1756`) exported as one 55,535,372-byte FP32 ONNX
policy. Its interface is `obs_dict [1, 1270] -> action [1, 23]`.

Full-reference MuJoCo results below compare the earlier step-7,400 policy with
step 25,450. Lower RMSE is better. The evaluation motions and videos are
separate, access-controlled artifacts and are intentionally not committed here.

| Evaluation length | Step 7,400 | Step 25,450 | Joint RMSE (rad) | Root position RMSE (m) | Root orientation RMSE (rad) |
|---:|---|---|---:|---:|---:|
| 51.84 s | completed | completed | 0.1581 -> **0.1388** | 0.6332 -> **0.4283** | **0.1534** -> 0.1535 |
| 64.06 s | completed | completed | 0.1357 -> **0.1040** | **0.3195** -> 0.3214 | 0.1194 -> **0.0686** |
| 108.22 s stress case | fell at 41.585 s | fell at **46.815 s** | 0.5394 -> **0.5164** | 1.8580 -> **1.0512** | **1.5943** -> 1.7124 |

On these three selected clips, the later checkpoint improves most tracking
measures and extends survival on the stress case by 5.23 seconds, but it does
**not** solve that motion. Stress-case RMSE includes the post-fall frames because
full-reference evaluation continues to the end. These are simulation
measurements, not claims of real-robot stability or safety.

Additional validated observations for this handoff:

- motion audit: 2,489 discovered -> 2,141 selected, with 233 quality failures
  and 115 additional SONIC filename-keyword exclusions;
- selected training data: 658,528 source frames, approximately 5.919 hours;
- measured 8 x RTX 4090 training throughput: approximately 282k timesteps/s
  after warm-up at 4,096 environments per GPU (32,768 total);
- measured memory in that short host-specific run: approximately 15.9 GiB per
  24 GiB GPU;
- Kengo asset contract: 23 actuated joints, 27 meshes, MuJoCo
  `(nq, nv, nu, nbody, nsensor) = (30, 29, 23, 25, 3)`;
- code/contract validation: 48 tests passed, ONNX checker and finite
  ONNX Runtime inference passed, and the dedicated sim2sim smoke test passed.

Training throughput and memory figures are measurements from one host, not
portable capacity guarantees.

## Implemented scope

```text
authorized retargeted NPZ + quality report
                    |
                    v
deterministic conversion + SONIC filtering + manifest
                    |
                    v
       Kengo 23-DoF SONIC PPO training
                    |
                    v
      hourly polling / one best checkpoint
                    |
                    v
      one combined 1270 -> 23 FP32 ONNX
                    |
                    v
      dedicated Kengo MuJoCo sim2sim
```

| Capability | Status |
|---|---|
| Kengo joint/body/order contract | Implemented and tested |
| Retargeted NPZ conversion | Implemented |
| Quality filtering and 40-keyword SONIC filter | Implemented and manifested |
| Multi-GPU tracker/controller training | Implemented; validated launcher targets 8 GPUs |
| Hourly best-checkpoint polling | Implemented; retains one full best checkpoint |
| Combined Kengo ONNX export | Implemented, `[1,1270] -> [1,23]` |
| Full-length MuJoCo sim2sim, metrics and recording | Implemented |
| Planner training for Kengo | Not available in the public upstream release |
| Native Kengo hardware transport and safety layer | **Not implemented** |

## Repository and artifact boundary

| Layer | Visibility | Contents |
|---|---|---|
| This parent repository | Public | Source, configuration, tests, documentation and one pinned gitlink |
| `I3NLi/kengo-robot-assets` | Private/restricted | Kengo URDF, MJCF, 27 STL meshes and their asset contract |
| Motion inputs and generated motion library | External/restricted | Retargeted NPZ, quality CSV, filtered PKL and manifest |
| Training/evaluation artifacts | External | PT/optimizer state, resolved config, ONNX, metrics and videos |

The private asset source states `Do Not distribute`. Access to the private
repository is an access-control mechanism, not a license grant. Do not expose
its contents through a public fork, Release, package, container layer, CI
artifact/cache, object store or experiment tracker. See
[KENGO_ASSET_BOUNDARY.md](KENGO_ASSET_BOUNDARY.md) for the complete boundary.

## Clone

Clone the public source without automatically downloading the upstream
repository's large Git LFS objects:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none \
  https://github.com/I3NLi/GEAR-SONIC-Kengo.git
cd GEAR-SONIC-Kengo
```

Pull individual upstream LFS paths only if you need them. The Kengo path does
not require downloading every upstream checkpoint and media object.

Authorized asset users must also have GitHub SSH access to the private asset
repository:

```bash
git submodule sync --recursive
git submodule update --init --recursive \
  external_dependencies/kengo_robot_description
python3 gear_sonic/scripts/stage_kengo_assets.py
```

Submodule initialization failing for an unauthorized public user is expected.
The parent pins asset tag `kengo-with-fist-v1.0.0` at commit
`6772bd3e8db9ad89cc3ee17c2dbbff7fe9f52771`; the validator fails closed if
the checkout, gitlink, lock file or asset contract disagree.

## Training environment

The validated training path targets Ubuntu 22.04+, Python 3.11, CUDA 12.x,
Isaac Sim 5.1 and Isaac Lab 2.3.x. Install Isaac Lab first using its
[official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
and provision that environment as `./.venv/bin/python`, which the launch scripts
use explicitly.

```bash
./.venv/bin/python -m pip install \
  -c gear_sonic/constraints-kengo-isaacsim-5.1.txt \
  -e "gear_sonic[training]" \
  pytest

./.venv/bin/python check_environment.py --training --robot kengo
```

For the full upstream environment procedure, see
[Installation (Training)](docs/source/getting_started/installation_training.md).

## Prepare filtered Kengo motions

The input path and quality report must be supplied from authorized storage.
The converter keeps `PASS` and `WARN`, rejects `FAIL`, applies the 40 public
SONIC filename keywords by default, and writes a schema-2 audit manifest beside
the output PKL.

```bash
./.venv/bin/python gear_sonic/data_process/convert_kengo_npz_to_motion_lib.py \
  --input /path/to/kengo-retargeted-raw \
  --quality-csv /path/to/per_file_quality.csv \
  --mjcf external_dependencies/kengo_robot_description/xml/kengo_with_fist.xml \
  --output data/kengo_motion_lib/robot_filtered/kengo_sonic_filtered.pkl
```

Keep the generated
`data/kengo_motion_lib/robot_filtered/kengo_sonic_filtered.pkl.manifest.json`
with the PKL. The 8-GPU launcher requires both files.

## Validate

```bash
./.venv/bin/python gear_sonic/scripts/stage_kengo_assets.py
./.venv/bin/python gear_sonic/scripts/validate_kengo_setup.py
./.venv/bin/python -m pytest -q \
  gear_sonic/tests/test_kengo_asset_submodule.py \
  gear_sonic/tests/test_kengo_contract.py \
  gear_sonic/tests/test_kengo_npz_conversion.py \
  gear_sonic/tests/test_kengo_sonic_sim2sim.py \
  gear_sonic/tests/test_backup_best_checkpoints.py \
  gear_sonic/tests/test_export_contract.py \
  gear_sonic/tests/test_im_eval_body_subsets.py \
  gear_sonic/tests/test_render_kengo_reference_video.py
```

The asset validator and the combined data validator are both necessary. Unit
tests using synthetic fixtures do not replace loading the pinned URDF/MJCF and
real mesh bundle.

## Train on the validated 8-GPU layout

```bash
bash gear_sonic/scripts/launch_kengo_training.sh \
  4096 100000 full8gpu_sonic_filtered

bash gear_sonic/scripts/status_kengo_training.sh 100
bash gear_sonic/scripts/launch_kengo_checkpoint_backup.sh
```

`4096` is the number of environments **per GPU**. This launcher is fixed to
CUDA devices 0-7 and starts eight processes; adapt and revalidate it before
using a different topology. Runtime logs go to `run_logs/`, while Hydra runs
and checkpoints are written below
`logs_rl/TRL_Kengo_Track/manager/universal_token/all_modes/`.

The backup service performs one immediate check and then polls once per hour.
It does not watch the directory continuously and does not accumulate hourly
checkpoint copies. It retains one `best_so_far.pt`, small metadata and an
improvement history under `checkpoint_backups/hourly_best/`.

## Export one Kengo ONNX

For a PT stored with the resolved `config.yaml` from the same training run:

```bash
bash gear_sonic/scripts/export_kengo_onnx.sh \
  /path/to/run/model_step_025450.pt
```

For the checkpoint selected by the hourly-best service, use its provenance-aware
entry point:

```bash
bash gear_sonic/scripts/export_latest_kengo_best_onnx.sh
```

The latter binds the original run configuration and validates the resulting
ONNX with the ONNX checker plus a finite ONNX Runtime inference. Its canonical
output is
`sim2sim_exports/kengo_best_step_<step>/model_step_<step>_kengo.onnx`.

The ONNX is inference-only. Continuing PPO training requires the PT optimizer
state, matching resolved config and complete run provenance.

## Run full-length MuJoCo sim2sim

A lightweight simulation environment can be installed separately from Isaac
Lab:

```bash
python3 -m venv .venv-sim
source .venv-sim/bin/activate
python -m pip install -e "gear_sonic[sim]"
python -m pip install "mujoco>=3.2,<4" "onnxruntime>=1.18,<2" imageio-ffmpeg
```

Run every reference frame, retain failure information, write metrics and record
the simulation:

```bash
python gear_sonic/scripts/run_kengo_sonic_sim2sim.py \
  --policy /path/to/model_step_025450_kengo.onnx \
  --motion /path/to/kengo_ref.npz \
  --xml external_dependencies/kengo_robot_description/xml/kengo_with_fist.xml \
  --headless --no-real-time --full-reference \
  --metrics-json /artifact/path/metrics.json \
  --video /artifact/path/sim2sim.mp4
```

`--full-reference` runs every reference frame even after a detected fall, but
the result remains failed and the process exits non-zero. For a short smoke
test, replace `--full-reference` with `--max-sim-seconds 4`.

Always keep the reference motion, model, metrics and video in an artifact store
whose access policy matches their provenance; do not add them to this repository.

## Important limitations

- This fork adapts the public SONIC tracker/controller training path, not the
  unreleased planner-training pipeline. The public planner artifact remains
  tied to the G1 interface.
- `gear_sonic_deploy`, Unitree DDS/C++, VLA/teleoperation and the legacy
  `run_sim_loop.py` path remain G1-specific. They must not command Kengo.
- Internal keys such as `g1`, `g1_dyn` and `g1_kin` are retained only as the
  state-dict ABI of existing Kengo checkpoints. Renaming them without a state
  translation breaks checkpoint compatibility.
- Passing asset validation or sim2sim proves structural simulator usability;
  it does not approve hardware operation.
- Kengo sim2real still requires an independently reviewed 23-motor transport
  and safety adapter covering index/sign/unit/zero mapping, limits, watchdog,
  emergency stop, fall handling, command arbitration and communication loss.

## Project map

| Path | Purpose |
|---|---|
| `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_kengo.yaml` | Kengo experiment configuration |
| `gear_sonic/utils/kengo_contract.py` | Joint, body, observation and action contract |
| `gear_sonic/data_process/convert_kengo_npz_to_motion_lib.py` | Deterministic conversion, filtering and manifest generation |
| `gear_sonic/scripts/launch_kengo_training.sh` | Validated eight-GPU launcher |
| `gear_sonic/scripts/backup_best_checkpoints.py` | Space-bounded hourly best-checkpoint selection |
| `gear_sonic/scripts/export_kengo_onnx.sh` | Combined Kengo policy export |
| `gear_sonic/scripts/run_kengo_sonic_sim2sim.py` | Dedicated MuJoCo sim2sim and recorder |
| `gear_sonic/tests/` | Contract, conversion, export, backup and sim2sim tests |

## Documentation

- [Kengo training handoff](KENGO_TRAINING.md)
- [Asset, artifact and sim2real boundary](KENGO_ASSET_BOUNDARY.md)
- [Contribution policy](CONTRIBUTING.md)
- [NVIDIA upstream documentation](https://nvlabs.github.io/GR00T-WholeBodyControl/)
- [GEAR-SONIC project page](https://nvlabs.github.io/GEAR-SONIC/)
- [GEAR-SONIC model page](https://huggingface.co/nvidia/GEAR-SONIC)
- [SONIC paper](https://arxiv.org/abs/2511.07820)

## Upstream, license and citation

This repository is a public GitHub fork of
[`NVlabs/GR00T-WholeBodyControl`](https://github.com/NVlabs/GR00T-WholeBodyControl).
The Kengo adaptation started from upstream revision
`a0732b642c0333077e127a2f56ab0014c196bca4`. It is not an official NVIDIA or
Galaxea release.

Repository source code remains under Apache License 2.0. NVIDIA-distributed
upstream model weights are governed by the NVIDIA Open Model License. The
private Kengo assets, Kengo motion data and Kengo-derived models are separate
materials and are **not** automatically licensed by either parent-repository
statement. Review [LICENSE](LICENSE), [`legal/`](legal/) and the asset boundary
before use or distribution.

If you use SONIC in research, cite the upstream work:

```bibtex
@article{luo2025sonic,
  title   = {SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control},
  author  = {Luo, Zhengyi and Yuan, Ye and Wang, Tingwu and Li, Chenran and Chen, Sirui and Castaneda, Fernando and Cao, Zi-Ang and Li, Jiefeng and Minor, David and Ben, Qingwei and Da, Xingye and Ding, Runyu and Hogg, Cyrus and Song, Lina and Lim, Edy and Jeong, Eugene and He, Tairan and Xue, Haoru and Xiao, Wenli and Wang, Zi and Yuen, Simon and Kautz, Jan and Chang, Yan and Iqbal, Umar and Fan, Linxi and Zhu, Yuke},
  journal = {arXiv preprint arXiv:2511.07820},
  year    = {2025}
}
```

Upstream components also build on
[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) and
[Isaac Lab](https://github.com/isaac-sim/IsaacLab). Use this repository's issue
tracker for Kengo-specific changes; reproduce upstream-only issues against the
NVIDIA repository before reporting them upstream.
