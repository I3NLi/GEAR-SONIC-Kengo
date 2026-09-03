# Kengo SONIC training handoff

This tree adapts the public NVIDIA GEAR-SONIC tracker/controller training code
to the 23-DoF Galaxea Kengo embodiment.  It does not claim to adapt the
unreleased planner-training pipeline; the public planner artifact is tied to
the G1 interface.

## Reproducible inputs

- Public NVIDIA base revision: `a0732b642c0333077e127a2f56ab0014c196bca4`
- Kengo fork revision: the parent commit being run; it must pin the reviewed
  `external_dependencies/kengo_robot_description` gitlink.
- Private Kengo robot assets: the pinned submodule, not a copied sibling tree.
- Raw retargeted input: an explicitly supplied, access-controlled directory.
- Kengo quality report: an explicitly supplied `per_file_quality.csv` with one
  row for every discovered input motion.
- Training motion library:
  `data/kengo_motion_lib/robot_filtered/kengo_sonic_filtered.pkl`
- Audit manifest:
  `data/kengo_motion_lib/robot_filtered/kengo_sonic_filtered.pkl.manifest.json`

The private submodule contains only robot-description assets. Motion data,
filtered PKL files, PT/ONNX models, metrics, and videos are separate artifacts;
see [KENGO_ASSET_BOUNDARY.md](KENGO_ASSET_BOUNDARY.md). Initialize and verify
the exact pinned asset commit before conversion, training, or sim2sim:

```bash
git submodule sync --recursive
git submodule update --init --recursive \
  external_dependencies/kengo_robot_description
python gear_sonic/scripts/stage_kengo_assets.py
```

The converter keeps `PASS` and `WARN`, rejects `FAIL`, and then applies the 40
public SONIC filename keywords from `filter_and_copy_bones_data.py`.  The
manifest records every keyword exclusion.  Current audited counts are 2489
discovered, 233 quality failures, 115 additional SONIC keyword exclusions, and
2141 selected motions / 658528 source frames (about 5.919 hours). These numbers
describe the audited input set; the full PKL is not committed to this repository.

Generate the full library deterministically from authorized local inputs:

```bash
python gear_sonic/data_process/convert_kengo_npz_to_motion_lib.py \
  --input /path/to/kengo-retargeted-raw \
  --quality-csv /path/to/per_file_quality.csv \
  --mjcf external_dependencies/kengo_robot_description/xml/kengo_with_fist.xml \
  --output data/kengo_motion_lib/robot_filtered/kengo_sonic_filtered.pkl
```

Retain the generated schema-2 manifest with each training run. It binds the
robot asset, quality filtering, SONIC keyword filtering, selected files, and
hashes; a bare PKL is not sufficient provenance.

## Validate

```bash
cd /path/to/GEAR-SONIC-Kengo
python gear_sonic/scripts/stage_kengo_assets.py
./.venv/bin/python gear_sonic/scripts/validate_kengo_setup.py
./.venv/bin/python -m pytest -q \
  gear_sonic/tests/test_kengo_contract.py \
  gear_sonic/tests/test_kengo_npz_conversion.py
```

## Launch and inspect

The measured stable full-size setting on the September 2026 eight-RTX-4090
training host was 4096 environments per GPU (32768 total). It used about
15.9 GiB of each 24 GiB GPU in the short benchmark and reached about 282k
timesteps/s after warm-up. This is a host-specific result, not a portable
default or a single-GPU capacity claim.

```bash
cd /path/to/GEAR-SONIC-Kengo
./gear_sonic/scripts/launch_kengo_training.sh 4096 100000 full8gpu_sonic_filtered
./gear_sonic/scripts/launch_kengo_checkpoint_backup.sh
./gear_sonic/scripts/status_kengo_training.sh 100
```

The launcher uses all eight GPUs, disables unsupported RTX 4090 NCCL P2P/IB,
runs detached with `setsid`/`nohup`, and records its PID and current stdout log
under `run_logs/`.  Hydra checkpoints are written below
`logs_rl/TRL_Kengo_Track/manager/universal_token/all_modes/`.

The checkpoint backup service wakes once per hour (with one immediate startup
check), scores only the then-current `last.pt` by its matching training-log
`Mean rewards`, and atomically updates `best_so_far.pt` when that sampled score
improves.  It does not continuously watch checkpoint writes and does not create
an accumulating hourly archive.  Exactly one full backup is retained, together
with SHA-256 metadata and a small JSONL improvement history under
`checkpoint_backups/hourly_best/`; the service PID and stdout log are under
`run_logs/`.

For restart/export reproducibility, archive the selected PT together with its
same-run resolved Hydra `config.yaml`, parent commit, asset-submodule commit,
filtered-data manifest, seed, and Isaac Sim/Isaac Lab/PyTorch/CUDA or container
versions. The ONNX alone cannot resume PPO training.

Isaac Sim may print Vulkan renderer errors in this container because it has no
usable graphics/Vulkan device.  Headless CUDA physics, PPO updates, DDP, and
checkpoint saves have nevertheless been verified.  This is not a substitute
for later visual replay on a Vulkan-capable host.

The public SONIC filename filter is semantic only.  Online adaptive sampling
continues to emphasize difficult segments during training.  A true closed-loop
trackability audit should be run over all 2141 motions after a mature Kengo
checkpoint exists; do not permanently remove clips based on one early model.

## Minimal ONNX export and local sim2sim

Kengo exports one combined 1270-to-23 FP32 policy by default. Its canonical artifact filename
is `model_step_<step>_kengo.onnx`; no encoder/decoder sidecars are needed by the
dedicated MuJoCo runner.

```bash
./gear_sonic/scripts/export_kengo_onnx.sh /path/to/best_so_far.pt

python gear_sonic/scripts/run_kengo_sonic_sim2sim.py \
  --policy /path/to/model_step_<step>_kengo.onnx \
  --headless --no-real-time --max-sim-seconds 4 \
  --metrics-json models/kengo_sonic_best/sim2sim_local_fp32_4s.json
```

The internal module keys `g1`, `g1_dyn`, and `g1_kin` are retained solely as
the checkpoint ABI used by the existing Kengo run.  Their observation and
action dimensions are Kengo's 23 DoFs; renaming those keys without translating
the state dict would invalidate trained checkpoints.

The upstream Unitree DDS/C++ deployment, VLA, teleoperation, and legacy MuJoCo
loop remain G1-specific and are intentionally isolated from this Kengo
sim2sim path.  Do not use them on Kengo hardware without implementing a native
23-motor state/command transport and safety layer.
