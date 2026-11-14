"""
E3 -- Gap-resolved phase diagram on the coin-angle slice (aA, aB).

Scans (aA, aB) in [-pi, pi]^2 at fixed (bA, bB) and computes at every point
  - the gap-resolved invariants (zeta_0, zeta_pi) from the half-step blocks,
  - both bulk gaps (from the smallest singular values of b and d),
  - the H_eff parity for both branch cuts (legacy quantity, cross-check only).

Audits printed:
  - quantization error of the half-zone integers on gapped points;
  - agreement with the closed form of Proposition 2 on gapped points;
  - GAP-SELECTIVE boundary audit: every zeta_0 jump between neighbouring grid
    points sits on a step where the 0 gap is small, every zeta_pi jump on a
    step where the pi gap is small;
  - identity: H_eff parity (either cut) = zeta_0 + zeta_pi + 1 (mod 2).

Outputs:
  ../figures/E3_phase_diagram.png
  ../figures/E3_phase_diagram.npz          (raw data)

Revision 2026-11-12: the first draft plotted the two branch-cut parities of
H_eff as "zeta_0" and "zeta_pi"; those are identical by an exact identity and
are not gap-resolved. They are stored here as ZH_pi / ZH_0 for the cross-check.
"""

import sys
import numpy as np
from figstyle import plt, INK, SEQ, BINARY
from spaceglide_walker import PI, gap_resolved_z2, closed_form_z2, point_data

BFIX = (-1.24, -1.39)
if len(sys.argv) == 3:                       # optional: another beta slice
    BFIX = (float(sys.argv[1]), float(sys.argv[2]))
TAG = "" if BFIX == (-1.24, -1.39) else f"_b{BFIX[0]:+.2f}_{BFIX[1]:+.2f}"
NG = 61
GAP_MIN = 0.05
grid = np.linspace(-PI, PI, NG)
Z0 = np.zeros((NG, NG), dtype=int)
ZP = np.zeros((NG, NG), dtype=int)
ZHpi = np.zeros((NG, NG), dtype=int)         # H_eff parity, branch cut at pi
ZH0 = np.zeros((NG, NG), dtype=int)          # H_eff parity, branch cut at 0
G0 = np.zeros((NG, NG))
GP = np.zeros((NG, NG))
QE = np.zeros((NG, NG))
CF = np.zeros((NG, NG), dtype=bool)          # closed form agrees

for i, aA in enumerate(grid):
    for j, aB in enumerate(grid):
        p = (aA, aB) + BFIX
        r = gap_resolved_z2(p, nk=161)
        Z0[i, j], ZP[i, j] = r["zeta0"], r["zetapi"]
        G0[i, j], GP[i, j], QE[i, j] = r["gap0"], r["gappi"], r["qerr"]
        CF[i, j] = (r["zeta0"], r["zetapi"]) == closed_form_z2(p)[:2]
        _, _, (zh_pi, _, _, _), (zh_0, _, _, _) = point_data(p, nk=161)
        ZHpi[i, j], ZH0[i, j] = zh_pi, zh_0
    if i % 10 == 0:
        print(f"  row {i + 1}/{NG}")

np.savez(f"../figures/E3_phase_diagram{TAG}.npz", grid=grid, Z0=Z0, ZP=ZP,
         ZHpi=ZHpi, ZH0=ZH0, G0=G0, GP=GP, QE=QE, BFIX=np.array(BFIX))

# --------------------------------------------------------------- audits
gapped = np.minimum(G0, GP) > GAP_MIN
print(f"\ngapped points (both gaps > {GAP_MIN}): {gapped.sum()} of {NG * NG}")
print(f"quantization error on gapped points: {QE[gapped].max():.2e}")
print(f"closed form agrees on gapped points: {CF[gapped].sum()} of {gapped.sum()}")
classes = sorted(set(zip(Z0[gapped].tolist(), ZP[gapped].tolist())))
print(f"classes (zeta_0, zeta_pi) present: {classes}")
ident = ((ZHpi == ZH0) & (ZHpi == (Z0 + ZP + 1) % 2))[gapped]
print(f"identity ZH(cut pi) = ZH(cut 0) = zeta_0 + zeta_pi + 1 on gapped "
      f"points: {ident.sum()} of {gapped.sum()}")

for name, Z, G, Gother in (("zeta_0", Z0, G0, GP), ("zeta_pi", ZP, GP, G0)):
    njump = bad = 0
    worst = 0.0
    other_at_jump = []
    for ax in (0, 1):
        dz = np.diff(Z, axis=ax) != 0
        own = np.minimum(np.take(G, range(0, NG - 1), axis=ax),
                         np.take(G, range(1, NG), axis=ax))
        oth = np.minimum(np.take(Gother, range(0, NG - 1), axis=ax),
                         np.take(Gother, range(1, NG), axis=ax))
        njump += dz.sum()
        bad += (dz & (own > 0.25)).sum()
        if dz.any():
            worst = max(worst, own[dz].max())
            other_at_jump.extend(oth[dz].tolist())
    print(f"{name}: {njump} jumps; largest OWN-gap floor on a jump step "
          f"{worst:.3f}; jumps across a step with own gap > 0.25: {bad}; "
          f"median OTHER gap on jump steps {np.median(other_at_jump):.3f}")

# --------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.3), constrained_layout=True)
ext = [-1, 1, -1, 1]                          # axes in units of pi
Y = np.cos((BFIX[0] + BFIX[1]) / 2)
cdb = np.cos((BFIX[0] - BFIX[1]) / 2)


def lines(ax, which):
    """Analytic gap-closing lines aA + aB = const (Proposition 2)."""
    xs = np.array([-1.0, 1.0])
    sets = []
    if which in ("0", "both") and abs(Y / cdb) < 1:
        S0 = 2 * np.arccos(Y / cdb)
        sets += [(s + 4 * PI * m, "-") for s in (S0, -S0) for m in (-1, 0, 1)]
        sets += [(2 * PI * m, ":") for m in (-1, 1)]       # Kramers, 0 gap
    if which in ("pi", "both") and abs(Y / cdb) < 1:
        Sp = 2 * np.arccos(-Y / cdb)
        sets += [(s + 4 * PI * m, "-") for s in (Sp, -Sp) for m in (-1, 0, 1)]
        sets += [(0.0, ":")]                               # Kramers, pi gap
    for S, ls in sets:
        ax.plot(xs, S / PI - xs, color=INK, lw=0.9, ls=ls)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)


for ax, Z, name, which in ((axes[0, 0], Z0, r"$\zeta_0$", "0"),
                           (axes[0, 1], ZP, r"$\zeta_\pi$", "pi")):
    ax.imshow(Z.T, origin="lower", extent=ext, cmap=BINARY, vmin=0, vmax=1,
              interpolation="nearest", aspect="auto")
    lines(ax, which)
    ax.set_title(f"{name}  (blue: 1, grey: 0)")
    ax.set_xlabel(r"$\alpha_A/\pi$")
    ax.set_ylabel(r"$\alpha_B/\pi$")
for ax, G, name, which in ((axes[1, 0], G0, r"gap $\Delta_0$", "0"),
                           (axes[1, 1], GP, r"gap $\Delta_\pi$", "pi")):
    im = ax.imshow(G.T, origin="lower", extent=ext, cmap=SEQ, vmin=0,
                   interpolation="bilinear", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(name)
    ax.set_xlabel(r"$\alpha_A/\pi$")
    ax.set_ylabel(r"$\alpha_B/\pi$")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
fig.savefig(f"../figures/E3_phase_diagram{TAG}.png")
print(f"saved ../figures/E3_phase_diagram{TAG}.png")
