"""
E12 -- Beyond the two-layer family.

E12a  Four-layer real half-step  F4 = S_dn C3 S_up C2 S_dn C1 S_up C0
      (eight coin angles). Nothing in Lemma 2 / Theorem 4 / Theorem 5 uses
      the two-layer form, so everything must carry over: det F = 1,
      quantization, Kramers pairs, parity = sign f(0), gap-selective jumps,
      and the wall rule with pinned-mode counts.
E12b  Are the Kramers-doubled closings phase boundaries?  In the two-layer
      family the both-gapped region of a given class (zeta_0, zeta_pi) splits
      into two pieces separated by the lines cos((aA+aB)/2) = +-1. We embed
      the family in a larger real, glide-chiral one,
          F' = F2(p) W,   W = exp(sum_i eta_i A_i) in SO(4) acting inside the
          unit cell (k-independent, real: K and T = Gamma K are kept),
      and search (multi-start optimisation over ten deformation parameters)
      for paths that connect the two pieces with BOTH gaps open. Finding one
      would show the Kramers closings are an accident of the two-layer family.
      Outcome: none found (see the printed summary); the question is open.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy.linalg import expm
from spaceglide_walker import (PI, rot, batched_F, batched_V, half_zone_integer,
                               closed_form_z2, gamma_real, shift_up_real,
                               shift_dn_real, coin_real, pinned_count)

results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def invariants_from_F(F, ks):
    """(zeta0, zetapi, qerr, gap0, gappi, f0, fpi) for a stack F(k) sampled on
    ks = linspace(0, 2 pi, nk) with both endpoints."""
    V = batched_V(ks)
    Ft = np.conj(np.transpose(V, (0, 2, 1))) @ F @ V
    b, d = Ft[:, :2, 2:], Ft[:, 2:, 2:]
    f0, fp = np.linalg.det(b), np.linalg.det(d)
    n0, q0 = half_zone_integer(f0)
    npi, qp = half_zone_integer(fp)
    sb = np.linalg.svd(b, compute_uv=False).min()
    sd = np.linalg.svd(d, compute_uv=False).min()
    return (n0 % 2, npi % 2, max(q0, qp), 2 * np.arcsin(min(sb, 1)),
            2 * np.arcsin(min(sd, 1)), f0, fp)


def F4(ks, q):
    """Four-layer real half-step; q = 8 angles (C0A, C0B, C1A, C1B, ...)."""
    Fa, _ = batched_F(ks, tuple(q[:4]))
    Fb, _ = batched_F(ks, tuple(q[4:]))
    return Fb @ Fa


rng = np.random.default_rng(404)
KS = np.linspace(0.0, 2 * PI, 257)

# ------------------------------------------------------------------- E12a
print("E12a -- four-layer real half-step (eight angles)")
tot = 0
qmax = detmax = kram = signok = 0.0
classes = {}
pts = []
for _ in range(400):
    q = rng.uniform(-PI, PI, 8)
    F = F4(KS, q)
    detmax = max(detmax, np.abs(np.linalg.det(F) - 1).max())
    z0, zp, qe, g0, gp, f0, fp = invariants_from_F(F, KS)
    if min(g0, gp) < 0.08:
        continue
    tot += 1
    qmax = max(qmax, qe)
    classes[(z0, zp)] = classes.get((z0, zp), 0) + 1
    signok += int(abs(f0[0].imag) < 1e-10 and abs(fp[0].imag) < 1e-10
                  and z0 == int(f0[0].real < 0) and zp == int(fp[0].real < 0))
    Fpi = F4(np.array([PI]), q)[0]
    t = np.array([[0, -1], [1, 0]], dtype=complex)
    G = np.kron(t, np.array([[0, 1], [1, 0]], dtype=complex))
    e = np.sort(-np.angle(np.linalg.eigvals(G @ Fpi.conj().T @ G.conj().T @ Fpi)))
    kram = max(kram, abs(e[0] - e[1]), abs(e[2] - e[3]))
    pts.append((q, (z0, zp), min(g0, gp)))
report("det F = 1; half-zone integers quantized", detmax < 1e-12 and qmax < 1e-12,
       f"{tot} gapped points, qerr {qmax:.1e}")
report("all four classes occur", len(classes) == 4, f"{dict(sorted(classes.items()))}")
report("parity = sign of f(0) (reality argument holds for any real F)",
       signok == tot, f"{int(signok)}/{tot}")
report("Kramers pairs at k = pi", kram < 1e-12, f"max splitting {kram:.1e}")

jumps = {"0": [], "pi": []}
for trial in range(40):
    q1, q2 = rng.uniform(-PI, PI, 8), rng.uniform(-PI, PI, 8)
    prev = None
    for s in np.linspace(0, 1, 161):
        cur = invariants_from_F(F4(KS[::2], (1 - s) * q1 + s * q2), KS[::2])
        if prev is not None:
            if cur[0] != prev[0]:
                jumps["0"].append((min(prev[3], cur[3]), min(prev[4], cur[4])))
            if cur[1] != prev[1]:
                jumps["pi"].append((min(prev[4], cur[4]), min(prev[3], cur[3])))
        prev = cur
for tag in ("0", "pi"):
    J = np.array(jumps[tag])
    report(f"zeta_{tag} jumps only where its own gap closes", J[:, 0].max() < 0.1,
           f"{len(J)} jumps on 40 random paths; own gap <= {J[:, 0].max():.3f}, "
           f"median other gap {np.median(J[:, 1]):.3f}")

# wall rule in the four-layer family (real space, sharp walls)
Nc = 60
Ns = 2 * Nc


def U4_wall(qI, qO):
    cell = np.arange(Ns) // 2
    inner = ((cell >= Nc // 4) & (cell < 3 * Nc // 4)).astype(float)
    par = np.arange(Ns) % 2
    layers = []
    for L in range(4):
        prof = np.where(par == 0, qO[2 * L], qO[2 * L + 1]) * (1 - inner) \
            + np.where(par == 0, qI[2 * L], qI[2 * L + 1]) * inner
        layers.append(coin_real(Ns, 0, 0, prof))
    Su, Sd, G = shift_up_real(Ns), shift_dn_real(Ns), gamma_real(Ns)
    F = Sd @ layers[3] @ Su @ layers[2] @ Sd @ layers[1] @ Su @ layers[0]
    return G @ F.T @ G.T @ F


good = sorted([p for p in pts if p[2] > 0.45], key=lambda p: -p[2])[:14]
tot = bad = 0
seen = set()
for i in range(len(good)):
    for j in range(i + 1, len(good)):
        dzz = tuple((a + b) % 2 for a, b in zip(good[i][1], good[j][1]))
        U = U4_wall(good[i][0], good[j][0])
        assert np.abs(U.imag).max() < 1e-13
        pc = pinned_count(U, Nc, tol=1e-4)
        tot += 1
        seen.add(dzz)
        bad += int(pc["0"][:2] != (dzz[0],) * 2 or pc["pi"][:2] != (dzz[1],) * 2)
report("wall rule: pinned modes per wall = (d zeta_0, d zeta_pi)", bad == 0,
       f"{tot} sharp walls between {len(good)} well-gapped points; "
       f"d zeta values covered {sorted(seen)}; mismatches {bad}")

# ------------------------------------------------------------------- E12b
print("E12b -- are the Kramers-doubled closings phase boundaries?")
gens = []
for i in range(4):
    for j in range(i + 1, 4):
        Aij = np.zeros((4, 4))
        Aij[i, j], Aij[j, i] = 1.0, -1.0
        gens.append(Aij)


def path_min_gap(p1, p2, x, nt=81, ks=KS[::4]):
    """Minimum over the path of min(gap0, gappi), and the set of classes met,
    for F'(t) = F2(p(t)) W(t) with
        p(t) = (1-t) p1 + t p2 + sin(pi t) x[6:10],
        W(t) = exp(sin(pi t) sum_i x[i] A_i),  i < 6.
    The end points are the unperturbed two-layer walks."""
    worst = 9.9
    cl = set()
    for t in np.linspace(0, 1, nt):
        s = np.sin(PI * t)
        p = tuple((1 - t) * np.array(p1) + t * np.array(p2) + s * np.asarray(x[6:10]))
        W = expm(s * sum(e * A for e, A in zip(x[:6], gens)))
        F, _ = batched_F(ks, p)
        z0, zp, qe, g0, gp, _, _ = invariants_from_F(F @ W[None], ks)
        worst = min(worst, g0, gp)
        cl.add((z0, zp))
    return worst, cl


BETA = (-1.24, -1.39)
cases = {
    "(0,1): aA+aB = +3.246 -> -3.246": ((2.094, 1.152) + BETA, (-1.152, -2.094) + BETA),
    "(1,1): aA+aB = +1.73  -> -1.73 ": ((2.827, -1.100) + BETA, (1.100, -2.827) + BETA),
    "(0,0): aA+aB = +4.82  -> -4.82 ": ((2.094, 2.723) + BETA, (-2.723, -2.094) + BETA),
}
from scipy.optimize import minimize


def objective(x, p1, p2):
    g, cl = path_min_gap(p1, p2, x, nt=41)
    return -g if len(cl) == 1 else 1.0 + len(cl)


found_all = True
for name, (p1, p2) in cases.items():
    assert closed_form_z2(p1)[:2] == closed_form_z2(p2)[:2]
    straight, _ = path_min_gap(p1, p2, np.zeros(10))
    best = (9.9, None)
    for start in range(12):                    # multi-start Nelder-Mead
        x0 = rng.normal(size=10) * np.r_[np.full(6, 1.2), np.full(4, 0.6)]
        res = minimize(objective, x0, args=(p1, p2), method="Nelder-Mead",
                       options=dict(maxiter=500, xatol=1e-3, fatol=1e-4))
        if res.fun < best[0]:
            best = (res.fun, res.x)
    gfine, cl = path_min_gap(p1, p2, best[1], nt=801, ks=KS)
    found = gfine > 0.1 and len(cl) == 1
    found_all = found_all and found
    print(f"    class {name}: straight path min gap {straight:.3f}; optimised "
          f"detour keeps both gaps >= {gfine:.3f} (801 path points, 257 momenta), "
          f"classes met {sorted(cl)}")
# This is an exploration, not a pass/fail check: its outcome is a finding.
print("  [INFO] gapped symmetry-preserving path between the two pieces of a "
      f"class found: {found_all}")
if not found_all:
    print("         No such path was found in a ten-parameter search (12 "
          "restarts per class): within this four-band real family the "
          "Kramers-doubled closings behave as genuine boundaries. Whether a "
          "finer invariant distinguishes the two pieces is open.")

print("\n" + ("E12 ALL PASS" if all(results) else ">>> E12 FAILURES <<<"))
