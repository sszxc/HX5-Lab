# Project
Sim, RL, teleop and device tools for a UR5 (CB3, not UR5e) arm and a HX5-D20-MRT-Right multi-fingered hand.
Real-robot runs go through `~/Documents/ros2_ur` (ROS 2 Jazzy via pixi); this repo talks to it over localhost UDP.

# Computing Environment
Runs locally. Python: `/home/lab/miniforge3/bin/python`. Headless rendering needs `MUJOCO_GL=egl`.

# Code Structure
Single user, agile development: refactor freely, no backwards compatibility needed.
Layout: `docs/architecture.md`; commands: `docs/usage.md`; task results and decisions: `docs/notes.md`.
