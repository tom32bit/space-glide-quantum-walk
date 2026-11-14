"""
E7 -- Defect experiments: what protects the wall modes, and how well.

E7a  Pinning mechanism of the pi-pinned wall modes of the A|B ring:
     - real angle disorder (wall-localised AND global) preserves the reality
       of the walk => antiunitary K (particle-hole) symmetry survives =>
       the modes stay pinned;
     - complex phase disorder breaks K => modes unpin linearly in strength.

E7b  Inter-wall hybridisation: deviation of the two pi modes from pi vs wall
     separation s on a large ring; delta ~ exp(-s/xi). Extracts xi.

E7c  S_pi signal optimisation: overlap of candidate local initial states with
     the exact pi modes (the T -> infinity value of S_pi), for the
     experimental-outlook estimate.

E7d  What exactly is protected?  A strictly local
     rank-one unitary  U -> U exp(i lam |m><m|),  |m> = the wall-1 pi mode
     truncated to +-10 cells, leaves both bulk domains untouched.
     - 0 < lam < pi breaks K: the mode leaves eps = pi, crosses the bulk band
       and is REMOVED from the gap -- existence is not protected by the glide
       invariant alone.
     - lam = pi is real (K-preserving) but not continuously connected to the
       identity within real unitaries: the mode is transferred from eps = pi
       to eps = 0. Only the total number of 0 and pi modes (mod 2) survives
       such non-gentle K-symmetric surgery; each gap separately is protected
       against continuous K-symmetric perturbations (E7a).

Outputs: figures/fig4_wall_protection.png, printed data blocks.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIGURES = ROOT / "figures"

import numpy as np
from figstyle import plt, C1, C2, C3, INK2, RULE
from spaceglide_walker import (PI, gamma_real, coin_real, shift_up_real,
                               shift_dn_real, bulk_gaps, two_wall_profiles,
                               U_profile, decay_rate)

P1 = (2.094395102393195, 2.7227136331111534, -1.24, -1.39)   # A: (0, 0)
P0 = (2.094395102393195, 1.1519173063162569, -1.24, -1.39)   # B: (0, 1)


def build_wall_U(Nc, pI, pO, w=3, c1=None, c2=None, angle_dis=None,
                 phase_dis=None, rng=None):
    """Ring with two walls; optional real angle disorder (mask, W) and complex
    phase disorder (mask, W). Returns U and the alpha profile."""
    Ns = 2 * Nc
    c1 = Nc // 4 if c1 is None else c1
    c2 = 3 * Nc // 4 if c2 is None else c2
    prof = np.zeros(Ns)
    for x in range(Ns):
        cell = x // 2
        s = 0.5 * (np.tanh((cell - c1) / w) - np.tanh((cell - c2) / w))
        aI = pI[0] if x % 2 == 0 else pI[1]
        aO = pO[0] if x % 2 == 0 else pO[1]
        prof[x] = aO + (aI - aO) * s
    if angle_dis is not None:
        mask, W = angle_dis
        prof = prof + np.where(mask, W * rng.uniform(-1, 1, Ns), 0.0)
    G = gamma_real(Ns)
    Su, Sd = shift_up_real(Ns), shift_dn_real(Ns)
    Rb = coin_real(Ns, pI[2], pI[3])
    Ra = coin_real(Ns, 0, 0, prof)
    Phi = np.eye(2 * Ns, dtype=complex)
    if phase_dis is not None:
        mask, W = phase_dis
        ph = np.where(mask, W * rng.uniform(-1, 1, Ns), 0.0)
        for x in range(Ns):
            Phi[2 * x + 1, 2 * x + 1] = np.exp(1j * ph[x])
    F = Sd @ Rb @ Su @ Phi @ Ra
    return G @ F.conj().T @ G.conj().T @ F, prof


def pi_deviation(U):
    eps = -np.angle(np.linalg.eigvals(U))
    return np.sort(PI - np.abs(eps))[:2]


# ------------------------------------------------------------------- E7a

print("E7a -- disorder study (K-symmetry mechanism)")
Nc = 60
Ns = 2 * Nc
rng = np.random.default_rng(42)
x = np.arange(Ns)
c1s, c2s = 2 * (Nc // 4), 2 * (3 * Nc // 4)
dwall = np.minimum.reduce([np.abs(x - c1s), np.abs(x - c2s),
                           Ns - np.abs(x - c1s), Ns - np.abs(x - c2s)])
wallmask = dwall <= 6
allmask = np.ones(Ns, bool)

Ws = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
NTRIAL = 40
curves = {}                     # tag -> array (len(Ws), NTRIAL) of deviations
for tag, kind in (("real, wall-localised", ("angle", wallmask)),
                  ("real, global", ("angle", allmask)),
                  ("complex phase, wall-localised", ("phase", wallmask))):
    devs = np.zeros((len(Ws), NTRIAL))
    for iw, W in enumerate(Ws):
        for it in range(NTRIAL):
            kw = {}
            if kind[0] == "angle":
                kw["angle_dis"] = (kind[1], W)
            else:
                kw["phase_dis"] = (kind[1], W)
            U, _ = build_wall_U(Nc, P1, P0, rng=rng, **kw)
            devs[iw, it] = pi_deviation(U).max()
        q10, q50, q90 = np.percentile(devs[iw], [10, 50, 90])
        print(f"  {tag:30s} W={W:.2f}: median {q50:.2e}  10-90% "
              f"[{q10:.1e}, {q90:.1e}]  max {devs[iw].max():.2e}")
    curves[tag] = devs
med_phase = np.median(curves["complex phase, wall-localised"], axis=1)
slope = np.polyfit(np.log(Ws[:5]), np.log(med_phase[:5]), 1)[0]
print(f"  K-breaking: median deviation ~ W^{slope:.2f} for W <= 0.3 "
      f"({NTRIAL} realisations per point)")
worst_real = max(curves["real, wall-localised"].max(),
                 curves["real, global"].max())
print(f"  K-preserving: largest deviation over all {2 * len(Ws) * NTRIAL} "
      f"real-disorder realisations (W up to 1.0): {worst_real:.2e}")

# ------------------------------------------------------------------- E7b

print("\nE7b -- inter-wall hybridisation vs separation (sharp walls)")
# Ring of 120 cells, inner domain A of length s between SHARP walls (smooth
# tanh walls add a profile-tail floor near 1e-8 that hides the decay beyond
# s ~ 28). The splitting is exp(-kappa_A s) with kappa_A the analytic decay
# rate of the pi-gap evanescent mode in domain A: the smallest |Im k| among
# the complex zeros of f_pi(k), see decay_rate(). Hybridisation through the
# outer domain, exp(-kappa_B (120 - s)), is negligible for s <= 32.
NcB = 120
seps = np.arange(6, 34, 2)
devs_sep = []
for s in seps:
    pa, pb = two_wall_profiles(NcB, P1, P0, w=0, c1=10, c2=10 + s)
    devs_sep.append(pi_deviation(U_profile(pa, pb)).max())
devs_sep = np.array(devs_sep)
coef = np.polyfit(seps, np.log(devs_sep), 1)
xi = -1.0 / coef[0]
kappa_A = decay_rate(P1, "pi")
print(f"  {len(seps)} separations, s = {seps.min()}..{seps.max()} cells, "
      f"splitting {devs_sep.max():.1e} -> {devs_sep.min():.1e}")
print(f"  fitted rate {-coef[0]:.4f} per cell (xi = {xi:.3f} cells), "
      f"r = {np.corrcoef(seps, np.log(devs_sep))[0, 1]:.7f}")
print(f"  analytic rate {kappa_A:.4f} per cell (xi = {1 / kappa_A:.3f} cells)")

# ------------------------------------------------------------------- E7c

print("\nE7c -- S_pi initial-state optimisation")
U, prof = build_wall_U(Nc, P1, P0)
evals, evecs = np.linalg.eig(U)
eps = -np.angle(evals)
modes = evecs[:, np.argsort(PI - np.abs(eps))[:2]]
Q, _ = np.linalg.qr(modes)                 # orthonormalise the two pi modes
wall_site = c1s                            # site index at wall 1


def spi_limit(psi0):
    return float(np.sum(np.abs(Q.conj().T @ psi0) ** 2))


cands = {}
v = np.zeros(2 * Ns, dtype=complex)
v[2 * wall_site] = 1
cands["|A up> at wall site"] = v
v = np.zeros(2 * Ns, dtype=complex)
v[2 * wall_site] = v[2 * wall_site + 1] = 1 / np.sqrt(2)
cands["equal coin superposition, one site"] = v
supp = [2 * wall_site, 2 * wall_site + 1,
        2 * (wall_site + 1), 2 * (wall_site + 1) + 1]
M = Q[supp, :] @ Q[supp, :].conj().T
w_, v_ = np.linalg.eigh(M)
v = np.zeros(2 * Ns, dtype=complex)
v[supp] = v_[:, -1]
cands["optimal state on the wall unit cell"] = v
supp = [2 * s_ + c for s_ in range(wall_site - 4, wall_site + 5) for c in (0, 1)]
M = Q[supp, :] @ Q[supp, :].conj().T
w_, v_ = np.linalg.eigh(M)
v = np.zeros(2 * Ns, dtype=complex)
v[supp] = v_[:, -1]
cands["optimal state on +-2 cells"] = v

for name, psi in cands.items():
    print(f"  {name:45s}: S_pi(inf) = {spi_limit(psi):.3f}")

# ------------------------------------------------------------------- E7d

print("\nE7d -- local rank-one surgery on the wall-1 pi mode")
g0_bulk = min(bulk_gaps(P1)[0], bulk_gaps(P0)[0])
gp_bulk = min(bulk_gaps(P1)[1], bulk_gaps(P0)[1])
xs = np.arange(Ns)
d1 = np.minimum(np.abs(xs - c1s), Ns - np.abs(xs - c1s))
d2 = np.minimum(np.abs(xs - c2s), Ns - np.abs(xs - c2s))
near1, near2 = d1 <= 20, d2 <= 20
mask1 = np.repeat(near1.astype(float), 2)
M2 = Q.conj().T @ (mask1[:, None] * Q)
wv2, vv2 = np.linalg.eigh(M2)
m = Q @ vv2[:, -1]                               # combination living on wall 1
m = np.where(np.repeat(near1, 2), m, 0.0)        # strictly local: +-10 cells
m = m * np.exp(-1j * np.angle(m[np.argmax(np.abs(m))]))
assert np.abs(m.imag).max() < 1e-6               # real up to a global phase
m = np.real(m) / np.linalg.norm(np.real(m))
print(f"  wall-1 mode: weight within 10 cells of the wall {wv2[-1]:.6f}")


def surgery_census(lam):
    Up = U @ (np.eye(2 * Ns) + (np.exp(1j * lam) - 1) * np.outer(m, m))
    ev, V = np.linalg.eig(Up)
    e = -np.angle(ev)
    ov = np.abs(V.conj().T @ m) ** 2 / np.sum(np.abs(V) ** 2, axis=0)
    j = np.argmax(ov)
    counts = []
    for dist, g in ((PI - np.abs(e), gp_bulk), (np.abs(e), g0_bulk)):
        idx = np.where(dist < 0.8 * g)[0]
        if len(idx) == 0:
            counts += [0.0, 0.0]
            continue
        Qw, _ = np.linalg.qr(V[:, idx])
        pr = (np.abs(Qw) ** 2).reshape(Ns, 2, -1).sum(axis=(1, 2))
        counts += [pr[near1].sum(), pr[near2].sum()]
    real = np.abs(Up.imag).max() < 1e-12
    return e[j], ov[j], counts, real


lams = np.linspace(0, PI, 41)
track = []
removed = []
print("  lam/pi  real   |eps_mode|/pi  overlap   pi-gap(w1,w2)   0-gap(w1,w2)")
for i, lam in enumerate(lams):
    e_m, o_m, cnt, real = surgery_census(lam)
    track.append(abs(e_m))
    n = [int(round(c)) for c in cnt]
    if n[0] == 0 and n[2] == 0:
        removed.append(lam)
    if i % 5 == 0:
        print(f"  {lam / PI:5.3f}   {str(real):5s}  {abs(e_m) / PI:8.4f}      "
              f"{o_m:6.3f}     ({n[0]}, {n[1]})          ({n[2]}, {n[3]})")
track = np.array(track)
print(f"  wall 1 hosts NO in-gap state for lam/pi in "
      f"[{min(removed) / PI:.3f}, {max(removed) / PI:.3f}] "
      f"(K broken, bulk untouched, wall 2 keeps its pi mode)")
print(f"  lam = pi (real): the mode sits at |eps| = {track[-1]:.1e}: it has "
      f"been moved from the pi gap to the 0 gap")

# ------------------------------------------------------------------- figure

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75), constrained_layout=True)

ax = axes[0]
styles = {"real, wall-localised": (C1, "o-", "real, at walls"),
          "real, global": (C3, "s-", "real, global"),
          "complex phase, wall-localised": (C2, "^-", "complex phase")}
for tag, dev in curves.items():
    c, st, lab = styles[tag]
    q10, q50, q90 = np.percentile(dev, [10, 50, 90], axis=1)
    ax.fill_between(Ws, q10, q90, color=c, alpha=0.2, lw=0)
    ax.loglog(Ws, q50, st, ms=4, color=c, markeredgecolor="white",
              markeredgewidth=0.6, label=lab)
ax.set_xlabel("disorder strength $W$")
ax.set_ylabel(r"$|\varepsilon-\pi|$ of the wall modes")
ax.legend(loc="lower left", fontsize=7)
ax.set_title(r"(a) $K$ pins; breaking $K$ unpins")

ax = axes[1]
ss = np.linspace(seps.min(), seps.max(), 200)
ax.semilogy(ss, devs_sep[0] * np.exp(-kappa_A * (ss - seps[0])), "-", lw=1.2,
            color=INK2, label=rf"analytic, $\xi={1 / kappa_A:.3f}$")
ax.semilogy(seps, devs_sep, "o", ms=4.5, color=C1, markeredgecolor="white",
            markeredgewidth=0.6, label="numerics")
ax.set_xlabel("wall separation $s$ (cells)")
ax.set_ylabel(r"splitting of the $\pi$ modes")
ax.legend(loc="upper right", fontsize=7)
ax.set_title("(b) inter-wall hybridisation")

ax = axes[2]
ax.axhspan(g0_bulk / PI, 1 - gp_bulk / PI, color=RULE, alpha=0.5, lw=0)
ax.text(0.04, 0.5, "bulk band", fontsize=7, color=INK2, va="center")
ax.text(0.30, 1 - 0.45 * gp_bulk / PI, r"$\pi$ gap", fontsize=7, color=INK2,
        va="center")
ax.text(0.55, 0.45 * g0_bulk / PI, "0 gap", fontsize=7, color=INK2,
        va="center")
ax.plot(lams / PI, track / PI, "-", color=C1)
ax.plot([0, 1], [track[0] / PI, track[-1] / PI], "o", ms=5, color=C1,
        markeredgecolor="white", markeredgewidth=0.8, zorder=5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.set_xlabel(r"surgery strength $\lambda/\pi$")
ax.set_ylabel(r"$|\varepsilon|/\pi$ of the wall mode")
ax.set_title("(c) local surgery removes it")

fig.savefig(FIGURES / "fig4_wall_protection.png")
print("\nsaved figures/fig4_wall_protection.png")
