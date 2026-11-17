"""Palm-tilt videos of the real-hand checkpoints at T = 4 s: one 3 x 3 grid per checkpoint (one panel per tilt, thumb
orange while it touches a ball) + per-tilt stats over --eps episodes (pace, drops, thumb-ball contact = fraction of
alive steps). Training-like physics + visual DR, same seeds for every checkpoint; the first episode of each tilt is filmed.
Run (base conda, one checkpoint per process): MUJOCO_GL=egl python -m robotics.policies.rl.baoding_tilt_videos dagger_rel
-> out/rl/baoding_real/videos/tilt_<ckpt>.mp4 / .txt"""
import argparse

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from robotics.paths import output_path
from robotics.policies.rl.baoding_dagger import load_student
from robotics.policies.rl.baoding_noisy_env import BaodingNoisyEnv, gravity_dir
from robotics.policies.rl.baoding_noisy_eval import set_cmd, set_tilt
from robotics.policies.rl.baoding_noisy_train import load_policy
from robotics.policies.rl.baoding_v3_videos import PANEL, T, View, panel, tilt_name, write

O = output_path("rl/baoding_real")
PERIOD = 4.0
# (tilt from palm-up, downhill azimuth) in deg; all inside the training cap (35 deg about the nominal 17 deg fingers-down)
TILTS = [(17, 0), (0, 0), (30, 0), (15, 90), (30, 90), (45, 0), (15, 270), (30, 270), (15, 180)]

p = argparse.ArgumentParser()
p.add_argument("ckpt", help="dir in out/rl/baoding_real: dagger_* (student.pkl) or teacher_* (params.pkl)")
p.add_argument("--eps", type=int, default=14, help="episodes per tilt")
a = p.parse_args()
n = len(TILTS) * a.eps
env = BaodingNoisyEnv(render=True, periods=(2.0, 16.0), warp=True, nworld=n, cam="wrist", tilt_max=np.radians(35),
                      tilt_about="nominal", dr=True, vis_dr=True, bg=str(O / "bg.npy"), stack=3, rel="_rel" in a.ckpt)
act = (load_policy(str(O / a.ckpt / "params.pkl"), False, env) if a.ckpt.startswith("teacher")
       else load_student(str(O / a.ckpt / "student.pkl")))

reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
s = reset(jax.random.split(jax.random.PRNGKey(0), n))
g = np.repeat([np.asarray(gravity_dir(np.radians(t), np.radians(ph))) for t, ph in TILTS], a.eps, 0)
s = set_cmd(set_tilt(env, s, jnp.asarray(g)), np.full(n, PERIOD))
film = np.arange(len(TILTS)) * a.eps
qpos, views, turn = [], [], np.zeros((T, n))
alive, drop_t, last = np.ones(n, bool), np.full(n, T), np.asarray(s.pipeline_state.qpos)
for t in range(T):
    views.append(np.asarray(s.obs["pixels/view_0"][film, ..., :3]))
    s = step(s, act(s.obs))
    last = np.where(alive[:, None], np.asarray(s.pipeline_state.qpos), last)  # freeze the scene once an episode ends
    qpos.append(last)
    turn[t] = np.where(alive, np.nan_to_num(np.asarray(s.metrics["turn"])), 0.0)
    ended = alive & (np.asarray(s.done) > 0)
    drop_t[ended], alive = t, alive & ~ended
qpos = np.stack(qpos)

m = env.mj_model
d, body = mujoco.MjData(m), m.geom_bodyid
thumb = {m.body(f"finger_r_link{k}").id for k in range(1, 5)}
balls = {m.body("ball0").id, m.body("ball1").id}


def touch(q):
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    return any({body[c.geom1], body[c.geom2]} & thumb and {body[c.geom1], body[c.geom2]} & balls for c in d.contact[:d.ncon])


tc = np.zeros((T, n), bool)
for i in range(n):
    for t in range(min(drop_t[i] + 1, T)):
        tc[t, i] = touch(qpos[t, i])
full = 2 * np.pi / PERIOD * 0.05 * T
names = [("17° fingers down (real pose)" if x == (17, 0) else tilt_name(*x)) for x in TILTS]
lines = [f"{a.ckpt}  T={PERIOD:.0f} s  {a.eps} eps/tilt  pace% / drop rate / thumb-ball contact% (alive steps)"]
for k, name in enumerate(names):
    e = slice(k * a.eps, (k + 1) * a.eps)
    live = np.arange(T)[:, None] <= drop_t[e][None]
    lines.append(f"  {name:30s} pace {100 * (turn[:, e].sum(0) / full).mean():4.0f}  drop {(drop_t[e] < T).mean():.2f}"
                 f"  thumb {100 * tc[:, e][live].mean():4.1f}")
out = O / "videos"
out.mkdir(exist_ok=True)
(out / f"tilt_{a.ckpt}.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)

v, short = View(env), a.ckpt.removeprefix("dagger_")


def panel_fn(k):
    i, cum = film[k], np.cumsum(turn[:, film[k]])

    def f(t):
        dropped, ta = drop_t[i] <= t, min(t, drop_t[i])
        sub = (f"pace {100 * cum[t] / (full * (t + 1) / T):3.0f}%  thumb on ball {100 * tc[:ta + 1, i].mean():3.0f}%"
               + ("  DROPPED" if dropped else ""))
        return panel(v(qpos[t, i], g[i], thumb if tc[t, i] else None), views[ta][k], f"{short} | {names[k]}", sub, dropped,
                     box=(6, PANEL - 86, 110, 80))  # bottom left: the thumb is on the right
    return f


fs = [panel_fn(k) for k in range(len(TILTS))]
write(str(out / f"tilt_{a.ckpt}.mp4"), [fs[:3], fs[3:6], fs[6:]])
