"""Printable Baoding ball matching the sim (R 20 mm, 60 g): two identical hemispheres with a central cavity for steel
shot + epoxy, aligned by a separate ring in an equator groove. Writes STLs (mm) to out/print/ and the shot count.
Run: python -m robotics.hardware.baoding_ball_print [--shot 5.0]"""
import argparse

import numpy as np

from robotics.paths import output_path

R, MASS = 20.0, 60.0          # mm, g; = robotics/policies/scripted/baoding.py R, MASS
R_CAV = 11.5                  # cavity: sphere up to 45 deg latitude, then a 45 deg cone (prints without support)
GROOVE = (14.55, 16.95, 2.0)  # alignment groove in each flat face: inner r, outer r, depth
CLEAR = 0.15                  # ring clearance per side
RHO = dict(pla=1.24, steel=7.85, epoxy=1.15)  # g/cm3; shell printed at 100% infill
N = 256                       # angular segments


def revolve(prof):
    """Closed CCW (r, z) profile (solid on the left; r=0 points on the axis) -> (F, 3, 3) triangles, outward normals."""
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ring = lambda p: np.stack([p[0] * np.cos(th), p[0] * np.sin(th), np.full(N, p[1])], 1)
    tris = []
    for p, q in zip(prof, np.roll(prof, -1, 0)):
        if p[0] == 0 and q[0] == 0:
            continue
        a, b = ring(p), ring(q)
        a1, b1 = np.roll(a, -1, 0), np.roll(b, -1, 0)
        if q[0] > 0:
            tris.append(np.stack([a, b1, b], 1))
        if p[0] > 0:
            tris.append(np.stack([a, a1, b1], 1))
    return np.concatenate(tris)


def half_profile():
    gi, go, gd = GROOVE
    t = np.linspace(0, np.pi / 2, 129)
    outer = np.stack([R * np.cos(t), R * np.sin(t)], 1)
    outer[-1, 0] = 0.0
    a = R_CAV / np.sqrt(2)
    s = np.linspace(np.pi / 4, 0, 33)[1:]
    cav = np.concatenate([[[0.0, R_CAV * np.sqrt(2)], [a, a]], np.stack([R_CAV * np.cos(s), R_CAV * np.sin(s)], 1)])
    return np.concatenate([outer, cav, [[gi, 0], [gi, gd], [go, gd], [go, 0]]])


def ring_profile():
    gi, go, gd = GROOVE
    h = 2 * (gd - CLEAR)
    return np.array([[gi + CLEAR, 0], [go - CLEAR, 0], [go - CLEAR, h], [gi + CLEAR, h]])


def write_stl(path, tri):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as f:
        f.write(b"baoding_ball_print".ljust(80, b"\0") + np.uint32(len(tri)).tobytes() + rec.tobytes())


def volume(tri):  # cm3
    return np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6000


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shot", type=float, nargs="+", default=[5.0, 4.5], help="steel shot diameters (mm)")
    args = p.parse_args()
    out = output_path("print")
    out.mkdir(parents=True, exist_ok=True)
    half, ring = revolve(half_profile()), revolve(ring_profile())
    write_stl(out / "baoding_half.stl", half)
    write_stl(out / "baoding_ring.stl", ring)
    v_half, v_ring, v_cav = volume(half), volume(ring), np.pi * R_CAV ** 3 / np.sqrt(2) / 1000  # cavity per half
    m_half, m_ring = RHO["pla"] * v_half, RHO["pla"] * v_ring
    print(f"{out}: half {len(half)} faces, {v_half:.2f} cm3, {m_half:.1f} g empty; ring {m_ring:.2f} g; "
          f"cavity {v_cav:.2f} cm3 per half")
    k = 1 / (1 - RHO["epoxy"] / RHO["steel"])  # steel replaces epoxy in the cavity
    m_steel = k * ((MASS - m_ring) / 2 - m_half - RHO["epoxy"] * v_cav)
    print(f"steel per half {m_steel:.1f} g (any shot size; +{k:.2f} g per g the empty half weighs below {m_half:.1f}), "
          f"fills {m_steel / RHO['steel'] / v_cav:.0%} of the cavity; then epoxy flush to the flat face")
    for d in args.shot:
        print(f"  = about {m_steel / (RHO['steel'] * np.pi / 6 * d ** 3 / 1000):.0f} shots of {d} mm")
