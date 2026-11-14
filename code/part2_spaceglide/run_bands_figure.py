"""
E2 -- Bulk bands and the half-step winding data.

Two columns:
  left  : real coins, the (zeta_0, zeta_pi) = (0, 1) representative B. The
          glide time reversal T = Gamma K (T^2 = e^{-ik}) pairs the bands at
          k = pi, 3 pi; f(2 pi - k) = f(k), so each trajectory is traced out
          and back and the parity is the sign of f(0).
  right : a generic SU(2)-coin walk (K and T broken, glide-chirality only).
          The Kramers pairs split; f(2 pi) = conj f(0) still holds and the
          half-zone parity is read off the open arc.
Rows:
  (a,b) quasienergy bands over the doubled zone k in [0, 4 pi];
  (c,d) lifted phases theta_0 = arg det b, theta_pi = arg det d over the
        doubled zone, showing theta(k + 2 pi) = -theta(k) (mod 2 pi);
  (e,f) trajectories of f_0 = det b and f_pi = det d for k in [0, 2 pi].

Output: ../figures/E2_bands_and_winding.png
"""

import numpy as np
from figstyle import plt, C1, C2, INK, INK2, RULE
from spaceglide_walker import (PI, gap_resolved_z2, su2_coin, batched_F,
                               U_from_F, halfstep_blocks)

B_REAL = (2.094395102393195, 1.1519173063162569, -1.24, -1.39)   # (0, 1)

# Deterministic generic SU(2) point in the same class (0, 1): among random
# draws, prefer well-gapped walks whose f_0(0), f_pi(0) are far from the real
# axis, so that the open arcs f(0) -> conj f(0) are clearly visible.
rng = np.random.default_rng(7)
best = None
for _ in range(3000):
    q = rng.normal(size=16)
    coins = tuple(su2_coin(q[4 * i:4 * i + 4]) for i in range(4))
    r = gap_resolved_z2(coins, nk=129)
    if (r["zeta0"], r["zetapi"]) != (0, 1):
        continue
    g = min(r["gap0"], r["gappi"])
    score = (min(g, 0.4) * abs(np.sin(np.angle(r["f0"][0])))
             * abs(np.sin(np.angle(r["fpi"][0]))))
    if best is None or score > best[0]:
        best = (score, coins, (r["zeta0"], r["zetapi"]), g)
GENERIC = best[1]
print(f"generic SU(2) point: class {best[2]}, min gap {best[3]:.3f}")

cases = [("real coins", B_REAL), ("generic SU(2) coins", GENERIC)]
fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.6), constrained_layout=True)

for col, (name, coins) in enumerate(cases):
    r = gap_resolved_z2(coins, nk=513)
    cls = (r["zeta0"], r["zetapi"])
    print(f"{name}: class {cls}, gaps ({r['gap0']:.3f}, {r['gappi']:.3f}), "
          f"f_0(0) = {r['f0'][0]:.4f}, f_pi(0) = {r['fpi'][0]:.4f}")

    ks = np.linspace(0, 4 * PI, 801)
    F, G = batched_F(ks, coins)
    eps = np.sort(-np.angle(np.linalg.eigvals(U_from_F(F, G))), axis=1)
    ax = axes[0, col]
    for b in range(4):
        ax.plot(ks / PI, eps[:, b] / PI, color=C1)
    for y in (-1, 0, 1):
        ax.axhline(y, color=RULE, lw=0.6)
    for x in (1, 3):
        ax.axvline(x, color=INK2, lw=0.6, ls=":")
    ax.set_xlim(0, 4)
    ax.set_ylim(-1.02, 1.02)
    ax.set_xlabel(r"$k/\pi$")
    ax.set_ylabel(r"$\varepsilon/\pi$")
    ax.set_title(f"{name}: bands, $(\\zeta_0,\\zeta_\\pi)=({cls[0]},{cls[1]})$")

    a, b, c, d = halfstep_blocks(ks, coins)
    f0, fpi = np.linalg.det(b), np.linalg.det(d)
    ax = axes[1, col]
    ax.plot(ks / PI, np.unwrap(np.angle(f0)) / PI, color=C1,
            label=r"$\theta_0=\arg\det b$")
    ax.plot(ks / PI, np.unwrap(np.angle(fpi)) / PI, color=C2,
            label=r"$\theta_\pi=\arg\det d$")
    ax.axvline(2, color=INK2, lw=0.6, ls=":")
    ax.set_xlim(0, 4)
    ax.set_xlabel(r"$k/\pi$")
    ax.set_ylabel(r"$\theta/\pi$")
    ax.legend(loc="best")
    ax.set_title(r"lifted phases over the doubled zone")

    half = ks <= 2 * PI + 1e-9
    ax = axes[2, col]
    ax.axhline(0, color=RULE, lw=0.6)
    ax.axvline(0, color=RULE, lw=0.6)
    for f, colr, lab in ((f0, C1, r"$f_0=\det b$"), (fpi, C2, r"$f_\pi=\det d$")):
        ax.plot(f[half].real, f[half].imag, color=colr, label=lab)
        ax.plot(f[0].real, f[0].imag, "o", ms=6, color=colr,
                markeredgecolor="white", markeredgewidth=1.0, zorder=5)
        kmid = np.argmin(np.abs(ks - 2 * PI))
        ax.plot(f[kmid].real, f[kmid].imag, "s", ms=6, color=colr,
                markeredgecolor="white", markeredgewidth=1.0, zorder=5)
    ax.plot(0, 0, "x", ms=6, color=INK)
    ax.set_xlabel(r"$\mathrm{Re}\,f$")
    ax.set_ylabel(r"$\mathrm{Im}\,f$")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    ax.set_title(r"$k\in[0,2\pi]$: circle $f(0)$, square $f(2\pi)=\overline{f(0)}$")

fig.savefig("../figures/E2_bands_and_winding.png")
print("saved ../figures/E2_bands_and_winding.png")
