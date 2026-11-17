# Project architecture

Sim, RL, teleop and device tools for a UR5 (CB3) + HX5-D20 right hand. Real-robot runs go
through `~/Documents/ros2_ur`; this repo talks to it over localhost UDP.

```text
HX5-Lab/
├── robotics/
│   ├── paths.py                    repo/asset/output paths, UR5 IP (env-var overrides)
│   ├── sim/                        MuJoCo scene composition, Sim runner, hand postures, viewer
│   ├── policies/scripted/          one script per hand task
│   ├── policies/rl/                MJX envs, brax PPO train/eval, Baoding vision DAgger + real-hand server
│   ├── teleop/                     Leap -> hand retargeting, record/replay
│   └── hardware/                   ur5/ ur_rtde examples; hx5/ real-hand UDP bridge (runs in the robotis_hand container) + sweep; realsense/ viewer
├── assets/mujoco/                  hand/arm/object MJCF (UR5e generic; lab/ = calibrated UR5 + mount + workcell from ros2_ur)
├── assets/teleop/                  Leap range-of-motion recordings (float32 npz, see teleop/leap_source.py)
├── ros2_ws/src/hx5_lab_scene/      UR5 Gazebo + MoveIt (see ros2_ws/CHEATSHEET.md)
├── tests/
└── docs/                           usage (commands), notes (task results, decisions), results (HTML reports)
```

Ignored locally: `out/` (outputs, checkpoints), `ros2_ws/{build,install,log}`, `old_ros_env_for_reference/`.

## Conventions

- Python `/home/lab/miniforge3/bin/python` (3.13): mujoco 3.7.0, mujoco-mjx 3.7.0, jax/jaxlib 0.11.2 (cuda12),
  brax 0.14.2, numpy 2.4.4, scipy 1.17.1, pytest 9.1.1, pynput 1.8.2, ur-rtde 1.6.5, leapc-cffi (`leap` 0.0.1).
  Install with `pip install --no-build-isolation --no-deps -e .`.
- `robotics/paths.py` env vars: `HX5LAB_REPO_ROOT`, `HX5LAB_OUTPUT_ROOT` (`out`), `HX5LAB_UR5_IP` (`192.168.1.132`).
- Git holds source, assets, configs, docs and small curated results; never checkpoints, videos, datasets or raw logs.
