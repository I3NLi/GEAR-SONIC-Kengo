# Kengo asset and runtime boundary

This document defines what the public-code parent project may consume from the
private Kengo robot-asset repository and what the asset repository must never
contain. It is an engineering control, not a grant of intellectual-property
rights.

## Repository responsibilities

### Parent: `GEAR-SONIC-Kengo`

The parent owns Kengo-specific source code and reproducible behavior:

- the 23-DoF joint/body/order contract;
- deterministic motion conversion and SONIC filename filtering;
- training, checkpoint-selection, ONNX export, validation, and sim2sim tools;
- configuration, tests, and documentation;
- one Git submodule pointer that pins the reviewed robot-asset commit.

The parent must not commit Kengo URDF, MJCF, or mesh bytes. It also must not
commit restricted motion data, quality reports, filtered PKL files,
checkpoints, ONNX files, reference clips, videos, credentials, or internal
operator logs. `.gitignore` is only a convenience and is not a security
boundary.

### Private submodule: `kengo-robot-assets`

The submodule at `external_dependencies/kengo_robot_description` owns only the
robot-description package:

- Kengo closed-fist URDF variants;
- the corresponding MuJoCo XML;
- 27 STL meshes;
- joint-name, ROS 2 package, launch, source-notice, and validation metadata.

It must not contain motion datasets, model artifacts, metrics, videos, secrets,
server addresses, or user-specific absolute paths. Its authoritative technical
contract is `ASSET_CONTRACT.json`; its Git commit is the immutable asset version
selected by the parent gitlink. `KENGO_ASSETS.lock.json` records the expected
repository, commit, contract digest, classification, and core counts; the
validator rejects disagreement between the lock, gitlink, and checkout.

### Data and model artifacts

Raw/retargeted motion, quality reports, filtered motion libraries, PT/optimizer
state, resolved training configuration, ONNX exports, evaluation metrics, and
videos are separate artifacts. They require their own provenance, access
policy, retention, and integrity manifest. Membership in either Git repository
does not grant rights to publish a dataset or trained model.

## Access and licensing boundary

The source asset package states `Do Not distribute`. The GitHub asset repository
must remain private, with least-privilege access only for users independently
authorized to use the assets. A private GitHub repository is an access-control
mechanism; it is not a license grant.

The source `package.xml` contains a `BSD` label, but the source package also
contains the conflicting distribution restriction. The stricter restriction
is applied until the rights holder provides written clarification. The parent
project's Apache-2.0 and NVIDIA model-license notices do not cover the private
submodule, Kengo motion data, or Kengo-derived models automatically.

Do not expose submodule contents through a public fork, release, package,
container layer, CI artifact/cache, public object store, telemetry service, or
experiment tracker. Private CI needs a dedicated read-only deploy key or token;
never place that credential in either repository.

## Clone and fail-closed behavior

An authorized checkout initializes the exact pinned asset commit:

```bash
git submodule sync --recursive
git submodule update --init --recursive \
  external_dependencies/kengo_robot_description
python gear_sonic/scripts/stage_kengo_assets.py
```

Despite its legacy filename, `stage_kengo_assets.py` no longer copies private
files. It verifies that the submodule is initialized at the pinned commit and
runs the asset repository's dependency-free validator. Kengo training,
conversion validation, and sim2sim must stop if that check fails; they must not
silently fall back to G1 assets or an unpinned sibling directory.

The runtime paths are:

```text
external_dependencies/kengo_robot_description/urdf/kengo_with_fist.urdf
external_dependencies/kengo_robot_description/xml/kengo_with_fist.xml
external_dependencies/kengo_robot_description/meshes/*.STL
```

`KENGO_SONIC_URDF` and `KENGO_SONIC_MJCF` may override these paths for explicit
testing. Overrides do not change the pinned production contract and must be
recorded in any resulting artifact manifest.

## Updating the pinned asset

Never depend on a moving `latest` reference or configure the parent to follow a
submodule branch automatically. To update:

1. change the private asset repository on a reviewed branch;
2. run `python scripts/validate_assets.py` there;
3. commit and push the immutable asset revision;
4. in the parent, fetch and check out that exact reviewed commit inside the
   submodule;
5. stage only the gitlink update and record the asset commit in the change;
6. run the Kengo contract tests, real MuJoCo asset load, ONNX validation, and a
   fixed-reference sim2sim regression;
7. roll back by restoring the previous parent gitlink if validation regresses.

A change to joint names/order, joint axes, units, root body, link frames,
geometry scale, actuator order, or sensor semantics is a breaking robot-contract
change. It requires a new contract version and must reject incompatible models.

## Simulation and hardware boundary

Passing asset validation or sim2sim proves only that the pinned model is
structurally usable in the tested simulator. It does not approve hardware use.
The upstream `gear_sonic_deploy`, Unitree DDS, VLA, teleoperation, and legacy
MuJoCo paths remain G1-specific and must not command Kengo.

Kengo sim2real requires a separately reviewed 23-motor transport and safety
layer, including motor-name/index/sign/unit/zero calibration, state estimation,
position/velocity/torque limits, watchdog, emergency stop, fall handling,
command arbitration, communication-loss behavior, HIL/suspended low-gain tests,
and an explicit hardware-safety approval. A model or asset commit alone never
authorizes real-robot actuation.
