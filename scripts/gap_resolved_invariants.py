"""
E9 -- Gap-resolved glide Z2 invariants: audits and the four-phase diagram.

E9a  Closed form (real coins, Proposition 2): parities and det b(0), det d(0)
     against the numerical half-zone integers at random points of the full
     four-angle family.
E9b  Relation to the Floquet operator's own index: the half-zone parity of
     det U_{+-} equals zeta_0 + zeta_pi (the integers themselves are defined
     only mod 2, through the branch of theta(0)), and the H_eff parity equals
     zeta_0 + zeta_pi + 1 (mod 2) for BOTH branch cuts, so branch cuts of
     H_eff cannot resolve the two gaps.
E9c  Glide-only protection: generic SU(2) coins in both layers (reality K and
     glide time reversal T = Gamma K both broken). Quantization, class census,
     and a gap-selective path audit: along random paths zeta_0 may jump only
     where the 0 gap closes, zeta_pi only where the pi gap closes.
E9d  Conventions: F -> iF (same U) flips both parities; the admissible frame
     change of `shifted_frame` flips zeta_0 only.
E9e  Kramers pairing at k = pi from T = Gamma K, T^2 = e^{-ik}: exact for real
     coins with cos eps(pi) = -cos((aA + aB)/2); lifted for generic coins.
E9f  Figure: all four classes in the plane (aA + aB, bA + bB) with the
     analytic phase boundaries cos(S_a/2) cos(D_b/2) = +/- cos(S_b/2).

Outputs: figures/fig2_phase_diagram.png, data/phase_diagram_plane.npz
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIGURES = ROOT / "figures"
DATA = ROOT / "data"

import numpy as np
from figstyle import (plt, C1, C2, INK, INK2, SEQ, CLASS_CMAP, CLASS_ORDER,
                      class_index)
from spaceglide_walker import (PI, gap_resolved_z2, closed_form_z2, su2_coin,
                               batched_F, U_from_F, batched_V, halfstep_blocks,
                               half_zone_integer, shifted_frame, U_bloch,
                               point_data)

GAP_MIN = 0.05
results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def U_parities(coins, nk=257):
    """Half-zone integers of det U_{+-} and of det (H_eff)_{+-} for both
    branch cuts, computed from U alone (no reference to F)."""
    ks = np.linspace(0.0, 2 * PI, nk)
    F, G = batched_F(ks, coins)
    U = U_from_F(F, G)
    V = batched_V(ks)
    VH = np.conj(np.transpose(V, (0, 2, 1)))
    nU, _ = half_zone_integer(np.linalg.det((VH @ U @ V)[:, :2, 2:]))
    ev, W = np.linalg.eig(U)
    eps = -np.angle(ev)
    Winv = np.linalg.inv(W)
    nH = []
    for cut in ("pi", "0"):
        e = eps if cut == "pi" else np.where(eps <= 0, eps + 2 * PI, eps)
        H = np.einsum("nij,nj,njk->nik", W, e, Winv)
        nH.append(half_zone_integer(np.linalg.det((VH @ H @ V)[:, :2, 2:]))[0])
    return nU, nH[0], nH[1]


# ------------------------------------------------------------------- E9a
print("E9a -- closed form vs numerics (real coins, random four-angle points)")
rng = np.random.default_rng(2026)
tot = ok = 0
worst = 0.0
classes = {}
for _ in range(600):
    p = tuple(rng.uniform(-2 * PI, 2 * PI, 4))
    r = gap_resolved_z2(p)
    if min(r["gap0"], r["gappi"]) < GAP_MIN:
        continue
    z0, zp, b0, d0 = closed_form_z2(p)
    tot += 1
    ok += int((r["zeta0"], r["zetapi"]) == (z0, zp))
    worst = max(worst, abs(r["f0"][0] - b0), abs(r["fpi"][0] - d0))
    classes[(z0, zp)] = classes.get((z0, zp), 0) + 1
report("parities match the closed form at every gapped point", ok == tot,
       f"{ok}/{tot}; classes {dict(sorted(classes.items()))}")
report("det b(0), det d(0) match the closed form", worst < 1e-12,
       f"max err {worst:.1e}")

# ------------------------------------------------------------------- E9b
print("E9b -- the U-based and H_eff-based parities are zeta_0 + zeta_pi (+1)")
tot = okU = okH = 0
for trial in range(240):
    if trial % 2 == 0:
        coins = tuple(rng.uniform(-PI, PI, 4))                  # real coins
    else:
        coins = tuple(su2_coin(rng.normal(size=4)) for _ in range(4))
    r = gap_resolved_z2(coins)
    if min(r["gap0"], r["gappi"]) < GAP_MIN:
        continue
    nU, nHpi, nH0 = U_parities(coins)
    tot += 1
    okU += int((nU - r["n0"] - r["npi"]) % 2 == 0)
    okH += int((nHpi - nU - 1) % 2 == 0 and (nH0 - nU - 1) % 2 == 0)
report("parity of det U_+- = zeta_0 + zeta_pi", okU == tot, f"{okU}/{tot}")
report("H_eff parity = zeta_0 + zeta_pi + 1 for BOTH branch cuts", okH == tot,
       f"{okH}/{tot}")

# ------------------------------------------------------------------- E9c
print("E9c -- generic SU(2) coins: glide-only protection")
tot = 0
qmax = 0.0
classes = {}
for _ in range(300):
    coins = tuple(su2_coin(rng.normal(size=4)) for _ in range(4))
    r = gap_resolved_z2(coins)
    if min(r["gap0"], r["gappi"]) < GAP_MIN:
        continue
    tot += 1
    qmax = max(qmax, r["qerr"])
    key = (r["zeta0"], r["zetapi"])
    classes[key] = classes.get(key, 0) + 1
report("quantization on gapped points", qmax < 1e-12,
       f"{tot} points, max err {qmax:.1e}")
report("all four classes realised", len(classes) == 4,
       f"{dict(sorted(classes.items()))}")

jumps = {"0": [], "pi": []}
for trial in range(60):
    q1, q2 = rng.normal(size=16), rng.normal(size=16)
    prev = None
    for s in np.linspace(0, 1, 121):
        q = (1 - s) * q1 + s * q2
        coins = tuple(su2_coin(q[4 * i:4 * i + 4]) for i in range(4))
        cur = gap_resolved_z2(coins, nk=129)
        if prev is not None:
            if cur["zeta0"] != prev["zeta0"]:
                jumps["0"].append((min(prev["gap0"], cur["gap0"]),
                                   min(prev["gappi"], cur["gappi"])))
            if cur["zetapi"] != prev["zetapi"]:
                jumps["pi"].append((min(prev["gappi"], cur["gappi"]),
                                    min(prev["gap0"], cur["gap0"])))
        prev = cur
for tag in ("0", "pi"):
    J = np.array(jumps[tag])
    report(f"zeta_{tag} jumps only where its OWN gap closes",
           J[:, 0].max() < 0.1,
           f"{len(J)} jumps on 60 random paths; largest own gap on a jump "
           f"step {J[:, 0].max():.3f}; median OTHER gap there "
           f"{np.median(J[:, 1]):.3f}")

# ------------------------------------------------------------------- E9d
print("E9d -- conventions")
tot = ok = 0
for _ in range(120):
    coins = tuple(su2_coin(rng.normal(size=4)) for _ in range(4))
    r = gap_resolved_z2(coins)
    if min(r["gap0"], r["gappi"]) < GAP_MIN:
        continue
    ri = gap_resolved_z2(coins, phase=1j)
    rf = gap_resolved_z2(coins, frame_fn=shifted_frame)
    tot += 1
    ok += int(ri["zeta0"] != r["zeta0"] and ri["zetapi"] != r["zetapi"]
              and rf["zeta0"] != r["zeta0"] and rf["zetapi"] == r["zetapi"])
report("F -> iF flips both; frame change flips zeta_0 only", ok == tot,
       f"{ok}/{tot}")
# The same common flip exists WITHIN real half-steps: F -> (1 x sigma_x) F is
# real, has det 1, gives the same U, and flips both parities. So reality of
# the coins does not make the pair a function of U alone.
SX4 = np.kron(np.eye(2), np.array([[0, 1], [1, 0]], dtype=complex))
ks_c = np.linspace(0.0, 2 * PI, 257)
V_c = batched_V(ks_c)
VH_c = np.conj(np.transpose(V_c, (0, 2, 1)))
tot = ok = 0
worst = 0.0
for _ in range(80):
    p = tuple(rng.uniform(-PI, PI, 4))
    r = gap_resolved_z2(p)
    if min(r["gap0"], r["gappi"]) < GAP_MIN:
        continue
    F, G = batched_F(ks_c, p)
    F2 = SX4[None] @ F
    worst = max(worst, np.abs(U_from_F(F2, G) - U_from_F(F, G)).max(),
                np.abs(np.linalg.det(F2) - 1).max(), np.abs(F2.imag).max()
                if np.abs(F.imag).max() == 0 else 0.0)
    Ft = VH_c @ F2 @ V_c
    n0, _ = half_zone_integer(np.linalg.det(Ft[:, :2, 2:]))
    npi, _ = half_zone_integer(np.linalg.det(Ft[:, 2:, 2:]))
    tot += 1
    ok += int(n0 % 2 != r["zeta0"] and npi % 2 != r["zetapi"])
report("real F -> (1 x sigma_x) F: same U, det 1, both parities flipped",
       ok == tot and worst < 1e-12, f"{ok}/{tot}, max err {worst:.1e}")

# ------------------------------------------------------------------- E9e
print("E9e -- Kramers pairing at k = pi (T = Gamma K, T^2 = e^{-ik})")
worst_split = worst_cf = 0.0
for _ in range(100):
    p = tuple(rng.uniform(-PI, PI, 4))
    e = np.sort(-np.angle(np.linalg.eigvals(U_bloch(PI, p))))
    worst_split = max(worst_split, abs(e[0] - e[1]), abs(e[2] - e[3]))
    worst_cf = max(worst_cf, np.abs(np.cos(e) + np.cos((p[0] + p[1]) / 2)).max())
report("real coins: bands pairwise degenerate at k = pi", worst_split < 1e-12,
       f"max splitting {worst_split:.1e}")
report("cos eps(pi) = -cos((aA + aB)/2)", worst_cf < 1e-12,
       f"max err {worst_cf:.1e}")
splits = []
for _ in range(100):
    coins = tuple(su2_coin(rng.normal(size=4)) for _ in range(4))
    F, G = batched_F(np.array([PI]), coins)
    e = np.sort(-np.angle(np.linalg.eigvals(U_from_F(F, G)[0])))
    splits.append(max(abs(e[0] - e[1]), abs(e[2] - e[3])))
report("generic SU(2) coins: degeneracy lifted", np.median(splits) > 0.05,
       f"median splitting {np.median(splits):.3f}")

# ------------------------------------------------------------------- E9g
print("E9g -- real coins: relation to the particle-hole (class D) Pfaffian")
# For real U(k), H_eff(k0) = i A(k0) with A real antisymmetric at k0 = 0, pi.
# The class-D parity sgn Pf A(0) Pf A(pi) of H_eff (branch cut at pi) must
# equal the H_eff glide parity, i.e. (-1)^(zeta_0 + zeta_pi + 1).
from scipy.linalg import logm


def pf4(A):
    return A[0, 1] * A[2, 3] - A[0, 2] * A[1, 3] + A[0, 3] * A[1, 2]


tot = ok = 0
for _ in range(400):
    p = tuple(rng.uniform(-PI, PI, 4))
    r = gap_resolved_z2(p)
    if min(r["gap0"], r["gappi"]) < 0.1:
        continue
    sgn = 1.0
    for k0 in (0.0, PI):
        A = np.real(logm(U_bloch(k0, p).real))
        sgn *= np.sign(pf4(A))
    tot += 1
    ok += int((0 if sgn > 0 else 1) == (r["zeta0"] + r["zetapi"] + 1) % 2)
report("Pfaffian parity of H_eff = zeta_0 + zeta_pi + 1 (mod 2)", ok == tot,
       f"{ok}/{tot} random real points")

# ------------------------------------------------------------------- E9f
print("E9f -- four-phase diagram in the (aA + aB, bA + bB) plane")
DA, DB = -0.6283185307179586, 0.9          # fixed differences aA - aB, bA - bB
NG = 81
S = np.linspace(-2 * PI, 2 * PI, NG)
CL = np.zeros((NG, NG), dtype=int)
GF = np.zeros((NG, NG))
QE = np.zeros((NG, NG))
agree = total = 0
for i, Sa in enumerate(S):
    for j, Sb in enumerate(S):
        p = ((Sa + DA) / 2, (Sa - DA) / 2, (Sb + DB) / 2, (Sb - DB) / 2)
        r = gap_resolved_z2(p, nk=161)
        CL[i, j] = class_index(r["zeta0"], r["zetapi"])
        GF[i, j] = min(r["gap0"], r["gappi"])
        QE[i, j] = r["qerr"]
        if GF[i, j] > GAP_MIN:
            total += 1
            agree += int((r["zeta0"], r["zetapi"]) == closed_form_z2(p)[:2])
np.savez(DATA / "phase_diagram_plane.npz", S=S, CL=CL, GF=GF, QE=QE,
         DA=DA, DB=DB)
report("plane scan: numerics match the closed form", agree == total,
       f"{agree}/{total} gapped points; quantization "
       f"{QE[GF > GAP_MIN].max():.1e}")
present = sorted(set(CL[GF > GAP_MIN].tolist()))
report("all four classes present in the plane", present == [0, 1, 2, 3],
       f"class indices {present}")

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), constrained_layout=True)
ext = [-2, 2, -2, 2]                       # axes in units of pi
ax = axes[0]
ax.imshow(CL.T, origin="lower", extent=ext, cmap=CLASS_CMAP, vmin=-0.5,
          vmax=3.5, interpolation="nearest", aspect="auto")
ss = np.linspace(-2 * PI, 2 * PI, 2001)
cdb = np.cos(DB / 2)
for sgn in (+1, -1):                       # cos(S_b/2) = sgn * cos(S_a/2) cdb
    y = 2 * np.arccos(np.clip(sgn * np.cos(ss / 2) * cdb, -1, 1))
    for branch in (y, -y):
        ax.plot(ss / PI, branch / PI, color=INK, lw=1.0)
for x0 in (-2, 0, 2):                      # parity-neutral Kramers closings
    ax.axvline(x0, color=INK, lw=0.9, ls=":")
for c in CLASS_ORDER:                      # direct label at the best-gapped point
    score = np.where(CL == class_index(*c), GF, -1.0)
    # keep the label inside the axes
    score[:6, :] = score[-6:, :] = score[:, :6] = score[:, -6:] = -1.0
    i, j = np.unravel_index(np.argmax(score), score.shape)
    ax.text(S[i] / PI, S[j] / PI, f"({c[0]},{c[1]})", ha="center",
            va="center", fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                      alpha=0.85))
ax.set_xlabel(r"$(\alpha_A+\alpha_B)/\pi$")
ax.set_ylabel(r"$(\beta_A+\beta_B)/\pi$")
ax.set_title(r"(a) classes $(\zeta_0,\zeta_\pi)$ and analytic boundaries")

ax = axes[1]
im = ax.imshow(GF.T, origin="lower", extent=ext, cmap=SEQ, vmin=0,
               interpolation="bilinear", aspect="auto")
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.set_label(r"$\min(\Delta_0,\Delta_\pi)$")
ax.set_xlabel(r"$(\alpha_A+\alpha_B)/\pi$")
ax.set_ylabel(r"$(\beta_A+\beta_B)/\pi$")
ax.set_title("(b) smaller of the two bulk gaps")
fig.savefig(FIGURES / "fig2_phase_diagram.png")
print("saved figures/fig2_phase_diagram.png")

print("\n" + ("E9 ALL PASS" if all(results) else ">>> E9 FAILURES <<<"))
