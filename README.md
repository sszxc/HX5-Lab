# HX5-Lab

MuJoCo simulation, scripted and RL dexterous tasks, Leap teleop and device tools for a
UR5 (CB3) arm with a ROBOTIS HX5-D20-MRT-Right hand. Real-robot runs go through `ros2_ur`.

- [Architecture](docs/architecture.md)
- [Commands](docs/usage.md)
- [Task results and notes](docs/notes.md)

```bash
/home/lab/miniforge3/bin/python -m pip install --no-build-isolation --no-deps -e .
/home/lab/miniforge3/bin/python -m pytest tests -q
```

History before 2026-11-16 lives in the Robot_Policy_Steering repo.
