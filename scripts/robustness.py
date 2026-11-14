"""
E11 -- Numerical robustness of the wall results.

E11a  Threshold sensitivity. The number of PINNED wall modes (eigenvalues
      within a tolerance of +1 or -1; no reference to the bulk gap) is
      compared with (d zeta_0, d zeta_pi) for all wall pairs, over a range of
      pinning tolerances and neighbourhood radii. Separately, the in-gap
      census of E4 is repeated over a range of window fractions and radii;
      its counts vary (unpinned pairs enter and leave the window) but its
      parities must not.
E11b  System size. Pinned counts and pinning error for rings of 40 to 200
      cells; the error must fall as exp(-kappa * Nc / 2) with the analytic
      rate kappa of the slower of the two domains.
E11c  Location of the phase boundaries. Bisection of the numerically computed
      invariants along random lines, compared with the analytic boundary.
E11d  Localisation length of the 0 mode of the B|C wall: analytic vs measured.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import itertools
import numpy as np
from spaceglide_walker import (PI, gap_resolved_z2, closed_form_z2, decay_rate,
                               U_profile, two_wall_profiles, pinned_count,
                               wall_census, bulk_gaps)

results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


BETA = (-1.24, -1.39)
REPS = {
    "A": (2.094395102393195, 2.7227136331111534) + BETA,
    "B": (2.094395102393195, 1.1519173063162569) + BETA,
    "C": (2.8274333882308134, -1.0995574287564276) + BETA,
    "D": (-0.6355, -2.4256, -3.0049, -3.1354),
    "C'": (0.6, 0.9) + BETA,
    "B'": (0.9, 2.1) + BETA,
    "A'": (-2.042, -2.5133) + BETA,
}
PAIRS = (list(itertools.combinations("ABCD", 2))
         + [("C", "C'"), ("B", "B'"), ("A", "A'")])
cls = lambda p: closed_form_z2(p)[:2]
dz = lambda a, b: tuple((x + y) % 2 for x, y in zip(cls(REPS[a]), cls(REPS[b])))
GAPS = {k: bulk_gaps(p) for k, p in REPS.items()}

# ------------------------------------------------------------------- E11a
print("E11a -- threshold sensitivity (ring of 80 cells, wall width 3)")
Nc = 80
rings = {}
for a, b in PAIRS:
    pa, pb = two_wall_profiles(Nc, REPS[a], REPS[b], w=3.0)
    rings[(a, b)] = U_profile(pa, pb)
bad = tot = 0
for (a, b), U in rings.items():
    for tol in (1e-6, 1e-5, 1e-4, 1e-3):
        for radius in (12, 24, 40):
            pc = pinned_count(U, Nc, tol=tol, radius=radius)
            tot += 1
            bad += int(pc["0"][:2] != (dz(a, b)[0],) * 2
                       or pc["pi"][:2] != (dz(a, b)[1],) * 2)
report("pinned-mode count per wall EQUALS (d zeta_0, d zeta_pi), not just mod 2",
       bad == 0, f"{tot} (pair, tolerance, radius) combinations, mismatches {bad}")
bad = tot = 0
spread = {}
for (a, b), U in rings.items():
    g0 = min(GAPS[a][0], GAPS[b][0])
    gp = min(GAPS[a][1], GAPS[b][1])
    seen = set()
    for frac in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        for radius in (16, 24, 40):
            c = wall_census(U, Nc, g0, gp, frac=frac, radius=radius)
            n = tuple(int(round(v)) for t in ("0", "pi") for v in c[t][:2])
            seen.add(n)
            tot += 1
            bad += int(not (n[0] % 2 == n[1] % 2 == dz(a, b)[0]
                            and n[2] % 2 == n[3] % 2 == dz(a, b)[1]))
    spread[f"{a}|{b}"] = sorted(seen)
report("in-gap census: parities independent of window fraction and radius",
       bad == 0, f"{tot} combinations, parity violations {bad}")
print("    raw census counts (N_0 w1, N_0 w2, N_pi w1, N_pi w2) seen per pair:")
for k, v in spread.items():
    print(f"      {k:6s}: {v}")

# ------------------------------------------------------------------- E11b
print("E11b -- system size")
ok = True
for a, b, gap in (("A", "B", "pi"), ("B", "C", "0"), ("A", "C", "pi"), ("A", "C", "0")):
    kap = min(decay_rate(REPS[a], gap), decay_rate(REPS[b], gap))
    sizes = np.array([40, 60, 80, 120, 160, 200])
    errs = []
    for N in sizes:
        pa, pb = two_wall_profiles(int(N), REPS[a], REPS[b], w=0)   # sharp walls
        U = U_profile(pa, pb)
        pc = pinned_count(U, int(N), tol=1e-2)
        ok = ok and pc[gap][:2] == (1, 1)
        eps = -np.angle(np.linalg.eigvals(U))
        d = np.abs(eps) if gap == "0" else PI - np.abs(eps)
        errs.append(np.sort(d)[:2].max())
    errs = np.array(errs)
    sel = errs > 1e-12
    rate = -np.polyfit(sizes[sel] / 2, np.log(errs[sel]), 1)[0] if sel.sum() >= 2 else np.nan
    print(f"    {a}|{b}, eps = {gap:2s}: pinning error {errs[0]:.1e} (40 cells) -> "
          f"{errs[sel][-1]:.1e} ({sizes[sel][-1]} cells); fitted rate per cell of "
          f"separation {rate:.3f}, analytic slower rate {kap:.3f}")
    ok = ok and abs(rate - kap) < 0.05 * kap
report("one pinned mode per wall at every size; error decays at the analytic rate",
       ok, "sizes 40-200 cells, sharp walls")

# ------------------------------------------------------------------- E11c
print("E11c -- location of the phase boundaries by bisection")
rng = np.random.default_rng(77)
worst = 0.0
nline = 0
while nline < 40:
    bA, bB = rng.uniform(-PI, PI, 2)
    Y, cd = np.cos((bA + bB) / 2), np.cos((bA - bB) / 2)
    if abs(Y / cd) > 0.95:
        continue
    key = ("zeta0", +1) if nline % 2 == 0 else ("zetapi", -1)
    S_exact = 2 * np.arccos(key[1] * Y / cd)             # boundary in aA + aB
    dA = rng.uniform(-1, 1)
    f = lambda S: gap_resolved_z2(((S + dA) / 2, (S - dA) / 2, bA, bB),
                                  nk=1025)[key[0]]
    lo, hi = S_exact - 0.3, S_exact + 0.3
    if f(lo) == f(hi):
        continue
    flo = f(lo)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if f(mid) == flo:
            lo = mid
        else:
            hi = mid
    worst = max(worst, abs(0.5 * (lo + hi) - S_exact))
    nline += 1
report("numerical jump of zeta_0 / zeta_pi sits on the analytic boundary",
       worst < 1e-9, f"40 random lines, largest distance {worst:.1e}")

# ------------------------------------------------------------------- E11d
print("E11d -- localisation length of the 0 mode (B|C wall, inner domain C)")
kap = decay_rate(REPS["C"], "0")
seps = np.arange(6, 40, 2)
dev = []
for s in seps:
    pa, pb = two_wall_profiles(120, REPS["C"], REPS["B"], w=0, c1=10, c2=10 + s)
    eps = -np.angle(np.linalg.eigvals(U_profile(pa, pb)))
    dev.append(np.sort(np.abs(eps))[:2].max())
dev = np.array(dev)
sel = dev > 1e-11
rate = -np.polyfit(seps[sel], np.log(dev[sel]), 1)[0]
report("measured decay rate of the 0-mode splitting equals the analytic rate",
       abs(rate - kap) < 0.01 * kap,
       f"fit {rate:.4f} vs analytic {kap:.4f} per cell over {sel.sum()} separations "
       f"({dev[sel].max():.0e} to {dev[sel].min():.0e})")

print("\n" + ("E11 ALL PASS" if all(results) else ">>> E11 FAILURES <<<"))
