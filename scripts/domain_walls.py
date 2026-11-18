"""
E4 -- Domain walls: gap-resolved bulk-defect correspondence (real coins).
E5 -- Bridge to Part I: the canonical (scalar) twist on the glide walker.
E6 -- Dynamical detection: plain and staggered return amplitudes.

E4 protocol. Ring of Nc cells with two smooth walls separating two bulk
parameter sets (all four coin angles interpolated). For every pair of the four
class representatives A = (0,0), B = (0,1), C = (1,1), D = (1,0) and for three
same-class controls, count the in-gap states bound to each wall separately in
the 0 gap and in the pi gap (projector-trace census, see wall_census).
Prediction tested:   N_0 = d zeta_0 (mod 2),   N_pi = d zeta_pi (mod 2).
In particular A|C has d zeta = (1, 1): one 0 mode AND one pi mode per wall,
although the Floquet operator's own index zeta_0 + zeta_pi is the same on
both sides.

E5 protocol. Even ring, canonical twist (-1 on seam-crossing hops): spectrum
must equal the Bloch spectrum at half-odd-integer cell momenta, and the
glide-chiral identity must still hold.

E6 protocol. Launch a single-site state at an A|C wall and at a control wall;
    S_0(T)  = |(1/T) sum_t        <psi0|U^t|psi0>|  ->  weight on eps = 0 modes,
    S_pi(T) = |(1/T) sum_t (-1)^t <psi0|U^t|psi0>|  ->  weight on eps = pi modes.

Outputs: figures/fig3_domain_walls.png, printed tables.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIGURES = ROOT / "figures"

import itertools
import numpy as np
from figstyle import plt, C1, C2, INK, INK2, RULE, SEQ
from spaceglide_walker import (PI, U_real, U_bloch, gamma_real, bulk_gaps,
                               gap_resolved_z2, closed_form_z2, wall_ring,
                               wall_census)

BETA = (-1.24, -1.39)
REPS = {
    "A": (2.094395102393195, 2.7227136331111534) + BETA,     # (0, 0)
    "B": (2.094395102393195, 1.1519173063162569) + BETA,     # (0, 1)
    "C": (2.8274333882308134, -1.0995574287564276) + BETA,   # (1, 1)
    "D": (-0.6355, -2.4256, -3.0049, -3.1354),               # (1, 0)
    "C'": (0.6, 0.9) + BETA,                                 # (1, 1) control
    "B'": (0.9, 2.1) + BETA,                                 # (0, 1) control
    "A'": (-2.042, -2.5133) + BETA,                          # (0, 0) control
}
CLS, GAPS = {}, {}
print("representatives: class (zeta_0, zeta_pi) [numerical | closed form], gaps")
for k, p in REPS.items():
    r = gap_resolved_z2(p)
    CLS[k] = (r["zeta0"], r["zetapi"])
    GAPS[k] = (r["gap0"], r["gappi"])
    assert CLS[k] == closed_form_z2(p)[:2]
    assert np.allclose(GAPS[k], bulk_gaps(p), atol=2e-3)
    print(f"  {k:2s}: {CLS[k]} | {closed_form_z2(p)[:2]}   gaps "
          f"({r['gap0']:.3f}, {r['gappi']:.3f})")

# --------------------------------------------------------------- E4
Nc = 60
PAIRS = (list(itertools.combinations("ABCD", 2))
         + [("C", "C'"), ("B", "B'"), ("A", "A'")])
print(f"\nE4 -- walls on a ring of {Nc} cells; per-wall in-gap state counts")
print("  pair    w  predicted(d0,dpi)   N_0(wall1,wall2)  N_pi(wall1,wall2)  "
      "pinning: max |eps| of 0 modes, max pi-|eps| of pi modes")
all_ok = True
rings = {}
for a, b in PAIRS:
    dz = ((CLS[a][0] + CLS[b][0]) % 2, (CLS[a][1] + CLS[b][1]) % 2)
    g0 = min(GAPS[a][0], GAPS[b][0])
    gp = min(GAPS[a][1], GAPS[b][1])
    for w in (2, 3, 5):
        U = wall_ring(Nc, REPS[a], REPS[b], w=w)
        assert np.abs(U.imag).max() < 1e-13                 # K-symmetric
        c = wall_census(U, Nc, g0, gp)
        n0 = tuple(int(round(x)) for x in c["0"][:2])
        npi = tuple(int(round(x)) for x in c["pi"][:2])
        ok = (all(n % 2 == dz[0] for n in n0)
              and all(n % 2 == dz[1] for n in npi))
        all_ok &= ok
        # the topological modes are the ones pinned at the gap centre
        pin0 = c["0"][2][:2 * dz[0]]
        pinp = c["pi"][2][:2 * dz[1]]
        print(f"  {a:2s}|{b:2s}  {w}  {dz}            {n0}            {npi}"
              f"            {max(pin0) if pin0 else float('nan'):.1e}  "
              f"{max(pinp) if pinp else float('nan'):.1e}   "
              f"[{'OK' if ok else 'MISMATCH'}]")
        if w == 3:
            rings[(a, b)] = (U, g0, gp, dz)
print(f"  gap-resolved parity rule holds for all "
      f"{3 * len(PAIRS)} wall configurations: {all_ok}")

# --------------------------------------------------------------- E5
print("\nE5 -- canonical cell twist on the glide walker (Part I bridge)")
NcB = 24
Ns = 2 * NcB
for twist in (False, True):
    U = U_real(Ns, REPS["A"], twist=twist)
    G = gamma_real(Ns, twist=twist)
    uerr = np.max(np.abs(U.conj().T @ U - np.eye(2 * Ns)))
    cerr = np.max(np.abs(G @ U @ G.conj().T - U.conj().T))
    ev = np.sort(np.angle(np.linalg.eigvals(U)))
    ms = np.arange(NcB)
    kms = (2 * ms + 1) * PI / NcB if twist else 2 * PI * ms / NcB
    bl = np.sort(np.angle(np.concatenate(
        [np.linalg.eigvals(U_bloch(k, REPS["A"])) for k in kms])))
    serr = np.max(np.abs(ev - bl))
    lat = "half-odd-integer" if twist else "integer"
    print(f"  twist={twist}: unitarity {uerr:.1e}, glide-chiral {cerr:.1e}, "
          f"spectrum == Bloch at {lat} momenta to {serr:.1e}")

# --------------------------------------------------------------- E6
print("\nE6 -- return amplitudes at an A|C wall (one 0 mode + one pi mode)")
U_t, g0_t, gp_t, _ = rings[("A", "C")]
U_c, g0_c, gp_c, _ = rings[("C", "C'")]
Ns = 2 * Nc
wall_site = 2 * (Nc // 4)


def wall_modes(U, g0, gp):
    """The 0 mode and the pi mode bound to wall 1 (orthonormalised, picked
    from the in-gap subspaces by maximal weight near wall 1)."""
    ev, V = np.linalg.eig(U)
    eps = -np.angle(ev)
    x = np.arange(Ns)
    d = np.minimum(np.abs(x - wall_site), Ns - np.abs(x - wall_site))
    mask = np.repeat((d <= 24).astype(float), 2)
    out = {}
    for tag, dist in (("0", np.abs(eps)), ("pi", PI - np.abs(eps))):
        idx = np.argsort(dist)[:2]                 # the two pinned modes
        Q, _ = np.linalg.qr(V[:, idx])
        M = Q.conj().T @ (mask[:, None] * Q)
        wv, vv = np.linalg.eigh(M)
        out[tag] = (Q @ vv[:, -1], dist[idx].max())
    return out


modes = wall_modes(U_t, g0_t, gp_t)
# best single-site (one site, one coin) launch state for seeing BOTH modes
w0 = np.abs(modes["0"][0]) ** 2
wp = np.abs(modes["pi"][0]) ** 2
best = int(np.argmax(np.minimum(w0, wp)))
print(f"  launch state: site {best // 2} (wall centre at site {wall_site}), "
      f"coin {'up' if best % 2 == 0 else 'down'}; overlaps with the wall "
      f"modes: 0 mode {w0[best]:.4f}, pi mode {wp[best]:.4f}")
print(f"  largest single-site overlaps: 0 mode {w0.max():.4f}, "
      f"pi mode {wp.max():.4f}")


def return_amplitudes(U, idx, T=20000):
    psi0 = np.zeros(2 * Ns, dtype=complex)
    psi0[idx] = 1.0
    psi = psi0.copy()
    acc0 = accp = 0.0 + 0.0j
    sign = 1.0
    trace = []
    for t in range(1, T + 1):
        psi = U @ psi
        sign = -sign
        amp = np.vdot(psi0, psi)
        acc0 += amp
        accp += sign * amp
        if t % 100 == 0:
            trace.append((t, abs(acc0) / t, abs(accp) / t))
    return np.array(trace)


tr_t = return_amplitudes(U_t, best)
tr_c = return_amplitudes(U_c, best)
print(f"  T = 20000: topological wall S_0 = {tr_t[-1, 1]:.4f}, "
      f"S_pi = {tr_t[-1, 2]:.4f}; control wall S_0 = {tr_c[-1, 1]:.1e}, "
      f"S_pi = {tr_c[-1, 2]:.1e}")

# --------------------------------------------------------------- figure
fig = plt.figure(figsize=(7.0, 5.6), constrained_layout=True)
top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 1.0])
panels = [(("A", "B"), r"A$|$B: $\Delta\zeta=(0,1)$"),
          (("B", "C"), r"B$|$C: $\Delta\zeta=(1,0)$"),
          (("A", "C"), r"A$|$C: $\Delta\zeta=(1,1)$"),
          (("C", "C'"), r"C$|$C$'$: $\Delta\zeta=(0,0)$")]
x = np.arange(Ns)
dw = np.minimum.reduce([np.abs(x - 2 * (Nc // 4)), Ns - np.abs(x - 2 * (Nc // 4)),
                        np.abs(x - 2 * (3 * Nc // 4)),
                        Ns - np.abs(x - 2 * (3 * Nc // 4))])
taxes = top.subplots(1, 4, sharey=True)
for ax, (pair, title) in zip(taxes, panels):
    U, g0, gp, dz = rings[pair]
    ev, V = np.linalg.eig(U)
    eps = -np.angle(ev)
    ww = (np.abs(V) ** 2).reshape(Ns, 2, -1).sum(axis=1)[dw <= 8].sum(axis=0)
    order = np.argsort(eps)
    sc = ax.scatter(np.arange(len(eps)), eps[order] / PI, c=ww[order], s=6,
                    cmap=SEQ, vmin=0, vmax=1, edgecolors=INK2, linewidths=0.15)
    for y in (g0 / PI, -g0 / PI, 1 - gp / PI, -(1 - gp / PI)):
        ax.axhline(y, color=RULE, lw=0.7)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title)
    ax.set_xlabel("state index")
taxes[0].set_ylabel(r"$\varepsilon/\pi$")
cb = top.colorbar(sc, ax=list(taxes), location="right", shrink=0.85, pad=0.01)
cb.set_label("weight near walls")

baxes = bottom.subplots(1, 3)
ax = baxes[0]
cells = np.arange(Ns) / 2.0
for tag, colr, ls_, lab in (("0", C1, "-", r"$\varepsilon=0$ mode"),
                            ("pi", C2, "--", r"$\varepsilon=\pi$ mode")):
    dens = (np.abs(modes[tag][0]) ** 2).reshape(Ns, 2).sum(axis=1)
    ax.plot(cells, dens, color=colr, ls=ls_, label=lab)
ax.axvline(Nc // 4, color=INK2, lw=0.6, ls=":")
ax.set_xlim(Nc // 4 - 10, Nc // 4 + 10)
ax.set_xlabel("cell")
ax.set_ylabel(r"$|\psi|^2$ per site")
ax.legend(loc="upper left", fontsize=7)
ax.set_title(r"A$|$C wall: bound modes")

for ax, (j, name) in zip(baxes[1:], ((1, r"$S_0(T)$"), (2, r"$S_\pi(T)$"))):
    ax.loglog(tr_t[:, 0], tr_t[:, j], color=C1, label=r"A$|$C wall")
    ax.loglog(tr_c[:, 0], tr_c[:, j], color=C2, ls="--",
              label=r"control C$|$C$'$")
    ax.set_xlabel("step $T$")
    ax.set_title(name)
    ax.legend(loc="lower left", fontsize=7)
fig.savefig(FIGURES / "fig3_domain_walls.png")
print("saved figures/fig3_domain_walls.png")
