"""
E10 -- Verification of the wall theorem (Theorem 5) and the surgery rule.

Theorem 5 (real coins): a wall between bulk parameters p_L, p_R with the gap
at mu open on both sides binds N_mu = zeta_mu(p_L) + zeta_mu(p_R) (mod 2)
states at quasienergy mu. The proof has four ingredients, each checked here:

E10a  Closed form of f_0(k), f_pi(k) at ALL momenta (quadratic in sin(k/2)),
      hence the complete gap-closing set: k = 0 on X = +-Y, and k = pi
      (Kramers-doubled) on cos((aA+aB)/2) = -+1. Nothing else.
E10b  Base case, EXACT integer arithmetic: at the sweet spots
      p+ = (pi, 0, 0, 0) and p- = (pi, 0, pi, pi) every coin is 1 or
      -i sigma_y, U is a signed permutation with U^2 = -1 in the bulk, and a
      sharp wall carries exactly one 4-cycle of sign +1, i.e. exactly one
      eigenvalue -1 and one eigenvalue +1.
E10c  Homotopy invariance: for fixed asymptotic parameters the parities do not
      depend on the wall profile (random profiles with random detours).
E10d  Period shifts: shifting both alpha (or both beta) angles by 2 pi on one
      side leaves the bulk walk unchanged and binds an even number of modes.
E10e  Elementary crossings: a wall across a k = 0 closing binds an odd number
      of pinned modes in that gap; across a Kramers closing an even number.
E10f  Surgery rule: U -> U W with W a random LOCAL REAL orthogonal matrix
      preserves both parities at that wall if det W = +1 and flips both if
      det W = -1 (finite-dimensional determinant argument).
E10g  Localisation length: analytic value from the complex zero of f_pi versus
      the measured decay of the inter-wall splitting, sharp walls.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from spaceglide_walker import (PI, gap_resolved_z2, closed_form_z2, f_closed,
                               decay_rate, U_profile, two_wall_profiles,
                               pinned_count, bulk_gaps, wall_census)

results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


rng = np.random.default_rng(1105)
BETA = (-1.24, -1.39)
A = (2.094395102393195, 2.7227136331111534) + BETA      # (0, 0)
B = (2.094395102393195, 1.1519173063162569) + BETA      # (0, 1)
C = (2.8274333882308134, -1.0995574287564276) + BETA    # (1, 1)
D = (-0.6355, -2.4256, -3.0049, -3.1354)                # (1, 0)
REPS = {"A": A, "B": B, "C": C, "D": D}
cls = lambda p: closed_form_z2(p)[:2]

# ------------------------------------------------------------------- E10a
print("E10a -- closed form of f_0(k), f_pi(k) and the complete closing set")
worst = 0.0
for _ in range(200):
    p = tuple(rng.uniform(-2 * PI, 2 * PI, 4))
    r = gap_resolved_z2(p, nk=65)
    ks = np.linspace(0, 2 * PI, 65)
    f0, fp = f_closed(ks, p)
    worst = max(worst, np.abs(f0 - r["f0"]).max(), np.abs(fp - r["fpi"]).max())
report("closed form matches the numerical determinants at all k", worst < 1e-12,
       f"200 random points, max err {worst:.1e}")
# gaps are bounded below away from the predicted closing set
viol = 0
for _ in range(4000):
    p = tuple(rng.uniform(-2 * PI, 2 * PI, 4))
    Cc = np.cos((p[0] + p[1]) / 2)
    z0, zp, b0, d0 = closed_form_z2(p)
    r = gap_resolved_z2(p, nk=129)
    # distance to the predicted closing sets of each gap
    d_0 = min(abs(b0), abs(1 + Cc))
    d_pi = min(abs(d0), abs(1 - Cc))
    if (r["gap0"] < 0.02 and d_0 > 0.05) or (r["gappi"] < 0.02 and d_pi > 0.05):
        viol += 1
report("no gap closing away from the predicted set", viol == 0,
       f"4000 random points, violations {viol}")

# ------------------------------------------------------------------- E10b
print("E10b -- exact base case at the sweet spots (integer arithmetic)")
I2i = np.eye(2, dtype=int)
Rpi = np.array([[0, -1], [1, 0]], dtype=int)
sxi = np.array([[0, 1], [1, 0]], dtype=int)


def int_walk(Ns, ca, cb):
    D2 = 2 * Ns
    Ca, Cb, Su, Sd, G = (np.zeros((D2, D2), dtype=int) for _ in range(5))
    for x in range(Ns):
        Ca[2*x:2*x+2, 2*x:2*x+2] = Rpi if ca[x] else I2i
        Cb[2*x:2*x+2, 2*x:2*x+2] = Rpi if cb[x] else I2i
        Su[2*((x+1) % Ns), 2*x] = 1
        Su[2*x+1, 2*x+1] = 1
        Sd[2*((x-1) % Ns)+1, 2*x+1] = 1
        Sd[2*x, 2*x] = 1
        y = (x + 1) % Ns
        G[2*y:2*y+2, 2*x:2*x+2] = sxi
    F = Sd @ Cb @ Su @ Ca
    return G @ F.T @ G.T @ F


def cycles(U):
    D2 = U.shape[0]
    assert (np.abs(U).sum(axis=0) == 1).all() and (np.abs(U).sum(axis=1) == 1).all()
    nxt = np.argmax(np.abs(U), axis=0)
    sgn = U[nxt, np.arange(D2)]
    seen = np.zeros(D2, bool)
    out = []
    for j in range(D2):
        if seen[j]:
            continue
        L, s, i, sites = 0, 1, j, set()
        while not seen[i]:
            seen[i] = True
            L += 1
            s *= sgn[i]
            sites.add(i // 2)
            i = nxt[i]
        out.append((L, int(s), sorted(sites)))
    return out


ok = True
detail = ""
for Nc in (8, 12, 20, 40):
    Ns = 2 * Nc
    x = np.arange(Ns)
    w1, w2 = Ns // 4, 3 * Ns // 4
    U = int_walk(Ns, (x % 2 == 0).astype(int), ((x >= w1) & (x < w2)).astype(int))
    cyc = cycles(U)
    bulk = [c for c in cyc if c[:2] == (2, -1)]
    wall = [c for c in cyc if c[:2] != (2, -1)]
    good = (len(wall) == 2 and all(c[:2] == (4, 1) for c in wall)
            and len(bulk) == (2 * Ns - 8) // 2)
    ok = ok and good
    detail = f"wall cycles {[(c[0], c[1], c[2]) for c in wall]} (Nc={Nc})"
report("sharp sweet-spot wall: bulk 2-cycles of sign -1, one 4-cycle of sign +1 "
       "per wall (=> exactly one eigenvalue -1 and one +1 per wall)", ok, detail)
PP, PM = (PI, 0.0, 0.0, 0.0), (PI, 0.0, PI, PI)
report("sweet spots lie in classes (0,1) and (1,0)",
       cls(PP) == (0, 1) and cls(PM) == (1, 0), f"{cls(PP)}, {cls(PM)}")

# ------------------------------------------------------------------- E10c
print("E10c -- homotopy invariance: parity depends on the endpoints only")
Nc = 60
Ns = 2 * Nc


def random_wall(pI, pO, rng):
    """Two-wall ring from pO to pI and back with random centres/widths, plus
    a smooth random detour of all four angles confined to the wall regions."""
    c1 = Nc // 4 + rng.integers(-3, 4)
    c2 = 3 * Nc // 4 + rng.integers(-3, 4)
    pa, pb = two_wall_profiles(Nc, pI, pO, w=rng.uniform(0.0, 5.0), c1=c1, c2=c2)
    cell = np.arange(Ns) // 2
    for prof in (pa, pb):
        for c in (c1, c2):
            amp = rng.uniform(-2.5, 2.5, 2)
            width = rng.uniform(1.0, 4.0)
            bump = np.exp(-((cell - c) / width) ** 2)
            prof += bump * np.where(np.arange(Ns) % 2 == 0, amp[0], amp[1])
    return U_profile(pa, pb)


tot = bad = 0
for a, b in (("A", "B"), ("B", "C"), ("A", "C"), ("B", "D"), ("A", "D"), ("C", "D")):
    dz = tuple((x + y) % 2 for x, y in zip(cls(REPS[a]), cls(REPS[b])))
    g0 = min(bulk_gaps(REPS[a])[0], bulk_gaps(REPS[b])[0])
    gp = min(bulk_gaps(REPS[a])[1], bulk_gaps(REPS[b])[1])
    for _ in range(25):
        U = random_wall(REPS[a], REPS[b], rng)
        pc = pinned_count(U, Nc, tol=min(1e-3, 0.02 * min(g0, gp)))
        tot += 1
        bad += int(not (pc["0"][0] % 2 == dz[0] == pc["0"][1] % 2
                        and pc["pi"][0] % 2 == dz[1] == pc["pi"][1] % 2))
report("150 random wall profiles (random widths incl. sharp, random detours)",
       bad == 0, f"parity violations {bad}/{tot}")

# ------------------------------------------------------------------- E10d
print("E10d -- period shifts (same bulk walk on both sides) bind even numbers")
bad = tot = 0
for name in "ABCD":
    p = np.array(REPS[name])
    for shift in ((2 * PI, 2 * PI, 0, 0), (0, 0, 2 * PI, 2 * PI),
                  (2 * PI, -2 * PI, 0, 0), (4 * PI, 0, 0, 0)):
        for w in (0.0, 3.0):
            pa, pb = two_wall_profiles(Nc, tuple(p + np.array(shift)), tuple(p), w=w)
            pc = pinned_count(U_profile(pa, pb), Nc)
            tot += 1
            bad += int(any(n % 2 for n in pc["0"][:2] + pc["pi"][:2]))
report("walls between a point and its 2 pi-shifted copies", bad == 0,
       f"odd counts in {bad}/{tot} configurations")

# ------------------------------------------------------------------- E10e
print("E10e -- elementary crossings")


def straddle(p, direction, eps):
    p = np.array(p)
    return tuple(p - eps * direction), tuple(p + eps * direction)


# k = 0 closings: X = Y (gap 0) and X = -Y (gap pi) along the alpha-sum direction
ok0 = okpi = okk = True
info = []
for trial in range(60):
    bA, bB = rng.uniform(-PI, PI, 2)
    Yv, cd = np.cos((bA + bB) / 2), np.cos((bA - bB) / 2)
    if abs(Yv / cd) > 0.9 or abs(Yv) < 0.15:
        continue
    for gap, sgn in (("0", +1), ("pi", -1)):
        S = 2 * np.arccos(sgn * Yv / cd)              # alpha_A + alpha_B on the line
        dA = rng.uniform(-1, 1)
        p0 = np.array([(S + dA) / 2, (S - dA) / 2, bA, bB])
        pL, pR = straddle(p0, np.array([1.0, 1.0, 0, 0]), 0.35)
        gL, gR = bulk_gaps(pL), bulk_gaps(pR)
        if min(gL + gR) < 0.12 or cls(pL) == cls(pR):
            continue
        # close to a transition the modes are wide: use a long ring and count
        # the in-gap states per wall with the projector census
        NcL = 160
        pa, pb = two_wall_profiles(NcL, pR, pL, w=2.0)
        cen = wall_census(U_profile(pa, pb), NcL, min(gL[0], gR[0]),
                          min(gL[1], gR[1]), frac=0.6, radius=60)
        pc = {t: tuple(int(round(v)) for v in cen[t][:2]) for t in ("0", "pi")}
        other = "pi" if gap == "0" else "0"
        good = (pc[gap][0] % 2 == 1 and pc[gap][1] % 2 == 1
                and pc[other][0] % 2 == 0 and pc[other][1] % 2 == 0)
        if gap == "0":
            ok0 = ok0 and good
        else:
            okpi = okpi and good
        info.append((gap, pc[gap][:2]))
report("across X = Y: odd number of pinned 0 modes, even number of pi modes", ok0,
       f"{sum(1 for g, _ in info if g == '0')} walls")
report("across X = -Y: odd number of pinned pi modes, even number of 0 modes", okpi,
       f"{sum(1 for g, _ in info if g == 'pi')} walls")
# Kramers closings: alpha-sum = 0 (pi gap) and 2 pi (0 gap)
nk = 0
kr_counts = set()
for trial in range(12):
    bA, bB = rng.uniform(-PI, PI, 2)
    dA = rng.uniform(-1, 1)
    for S0 in (0.0, 2 * PI):
        p0 = np.array([(S0 + dA) / 2, (S0 - dA) / 2, bA, bB])
        pL, pR = straddle(p0, np.array([1.0, 1.0, 0, 0]), 0.5)
        gL, gR = bulk_gaps(pL), bulk_gaps(pR)
        if cls(pL) != cls(pR) or min(gL + gR) < 0.12:
            continue
        NcL = 160
        pa, pb = two_wall_profiles(NcL, pR, pL, w=2.0)
        cen = wall_census(U_profile(pa, pb), NcL, min(gL[0], gR[0]),
                          min(gL[1], gR[1]), frac=0.6, radius=60)
        counts = [int(round(v)) for t in ("0", "pi") for v in cen[t][:2]]
        okk = okk and all(n % 2 == 0 for n in counts)
        kr_counts.add(tuple(counts))
        nk += 1
report("across a Kramers-doubled closing: even number of states in each gap",
       okk, f"{nk} walls; (N_0, N_0, N_pi, N_pi) per wall seen: {sorted(kr_counts)}")

# ------------------------------------------------------------------- E10f
print("E10f -- surgery rule for random local real orthogonal W")
pa, pb = two_wall_profiles(Nc, A, C, w=3.0)          # A|C: one 0 mode + one pi mode
U0 = U_profile(pa, pb).real
base = pinned_count(U0, Nc)
w1 = 2 * (Nc // 4)
stats = {+1: [0, 0], -1: [0, 0]}
for trial in range(200):
    nloc = rng.integers(2, 9)                         # W acts on nloc basis states
    sites = rng.choice(np.arange(2 * (w1 - 6), 2 * (w1 + 7)), size=nloc, replace=False)
    Q, R = np.linalg.qr(rng.normal(size=(nloc, nloc)))
    Q = Q * np.sign(np.diag(R))
    det = int(round(np.linalg.det(Q)))
    W = np.eye(2 * Ns)
    W[np.ix_(sites, sites)] = Q
    pc = pinned_count(U0 @ W, Nc, tol=1e-6)
    flip0 = (pc["0"][0] - base["0"][0]) % 2
    flipp = (pc["pi"][0] - base["pi"][0]) % 2
    far = (pc["0"][1] == base["0"][1]) and (pc["pi"][1] == base["pi"][1])
    expected = 0 if det == 1 else 1
    stats[det][0] += 1
    stats[det][1] += int(flip0 == expected and flipp == expected and far)
report("det W = +1: both parities at the wall unchanged", stats[1][0] == stats[1][1],
       f"{stats[1][1]}/{stats[1][0]}")
report("det W = -1: both parities at the wall flipped (far wall unaffected)",
       stats[-1][0] == stats[-1][1], f"{stats[-1][1]}/{stats[-1][0]}")

# ------------------------------------------------------------------- E10g
print("E10g -- localisation length: analytic (complex zero of f_pi) vs measured")
kap_A, kap_B = decay_rate(A, "pi"), decay_rate(B, "pi")
print(f"  analytic decay rates of the pi mode: into A {kap_A:.4f}/cell "
      f"(xi = {1 / kap_A:.3f}), into B {kap_B:.4f}/cell (xi = {1 / kap_B:.3f})")
# Ring of 120 cells, inner domain A of length s between SHARP walls. The two
# pi modes hybridise through A as exp(-kappa_A s) and through the outer domain
# B as exp(-kappa_B (120 - s)); for s <= 32 the first dominates by more than
# four decades, and the splitting stays above the eigenvalue round-off floor.
NcB = 120
seps = np.arange(6, 34, 2)
dev = []
for s in seps:
    pa, pb = two_wall_profiles(NcB, A, B, w=0, c1=10, c2=10 + s)
    eps = -np.angle(np.linalg.eigvals(U_profile(pa, pb)))
    dev.append(np.sort(PI - np.abs(eps))[:2].max())
dev = np.array(dev)
sel = dev > 1e-11
coef = np.polyfit(seps[sel], np.log(dev[sel]), 1)
r = np.corrcoef(seps[sel], np.log(dev[sel]))[0, 1]
print(f"  sharp walls, {sel.sum()} separations from {seps[sel].min()} to "
      f"{seps[sel].max()} cells, splitting from {dev[sel].max():.1e} down to "
      f"{dev[sel].min():.1e}")
print(f"  fitted rate {-coef[0]:.4f}/cell (xi = {-1 / coef[0]:.3f}), r = {r:.7f}")
report("measured decay rate equals the analytic rate of the inner domain",
       abs(-coef[0] - kap_A) < 0.01 * kap_A,
       f"fit {-coef[0]:.4f} vs analytic {kap_A:.4f} "
       f"({100 * abs(-coef[0] - kap_A) / kap_A:.2f}% apart)")

print("\n" + ("E10 ALL PASS" if all(results) else ">>> E10 FAILURES <<<"))
