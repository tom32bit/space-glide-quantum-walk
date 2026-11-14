"""
E1 -- Construction verification for the space-glide walker.

  E1a  Unitarity (Bloch and real space), locality (banded real-space U).
  E1b  Real-space vs Bloch cross-validation (momentum-sector projection).
  E1c  Glide-chiral identity Gamma(k) U(k) Gamma(k)^{-1} = U(k)^dag, all k.
  E1d  2 pi periodicity of U(k); projectivity Gamma(k)^2 = e^{-ik} I.
  E1e  H_eff block-off-diagonal in the twisted basis (diagonal leak ~ 0).
  E1f  NO-COLLAPSE 1: no k-independent chiral operator Gamma_0 exists with
       Gamma_0 U(k) Gamma_0^{-1} = U(k)^dag  (else ordinary AIII in disguise).
  E1g  NO-COLLAPSE 2: the commutant of {U(k)} over sampled k is trivial
       (no k-independent block decomposition; the failure mode of the old
       Option B candidate constructions).
  E1h  Gamma-eigenbasis Moebius transition: V(k+2pi) = V(k) X with X the
       sector swap, and d(k+2pi) related to conj(d(k)) as derived.
"""

import sys
import numpy as np
from spaceglide_walker import (PI, U_bloch, U_real, F_real, gamma_real,
                               gamma_bloch, twisted_basis, heff_bloch,
                               offdiag_det, translate_real)

results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


params = (0.9, 2.1, 0.4, 1.3)      # generic point (aA, aB, bA, bB)
Nc = 12                            # cells -> 24 sites, dim 48
Ns = 2 * Nc

print("E1a -- unitarity and locality")
U = U_real(Ns, params)
err = np.max(np.abs(U.conj().T @ U - np.eye(2 * Ns)))
report("real-space unitarity", err < 1e-12, f"err={err:.2e}")
ks = np.linspace(0, 2 * PI, 13)
err = max(np.max(np.abs(U_bloch(k, params).conj().T @ U_bloch(k, params) - np.eye(4)))
          for k in ks)
report("Bloch unitarity", err < 1e-12, f"err={err:.2e}")
# locality: U should not connect sites farther than 4 sites apart (F range 2,
# conjugated F range 2, glide 1)
maxrange = 0
for i in range(2 * Ns):
    for j in range(2 * Ns):
        if abs(U[i, j]) > 1e-12:
            dx = abs(i // 2 - j // 2)
            dx = min(dx, Ns - dx)
            maxrange = max(maxrange, dx)
report("locality (hop range = 2 sites, i.e. one unit cell)", maxrange == 2,
       f"range={maxrange}")

print("E1b -- real space vs Bloch")
worst = 0.0
for m in range(Nc):
    k = 2 * PI * m / Nc
    # momentum-sector basis vectors |k, s, c>
    B = np.zeros((2 * Ns, 4), dtype=complex)
    for n in range(Nc):
        ph = np.exp(1j * k * n) / np.sqrt(Nc)
        B[2 * (2 * n) + 0, 0] = ph      # A up
        B[2 * (2 * n) + 1, 1] = ph      # A down
        B[2 * (2 * n + 1) + 0, 2] = ph  # B up
        B[2 * (2 * n + 1) + 1, 3] = ph  # B down
    Uk_num = B.conj().T @ U @ B
    leak = np.linalg.norm(U @ B - B @ Uk_num)
    worst = max(worst, leak, np.max(np.abs(Uk_num - U_bloch(k, params))))
report("sector-projected U == analytic U(k)", worst < 1e-12, f"err={worst:.2e}")

print("E1c -- glide-chiral identity")
err = max(np.max(np.abs(gamma_bloch(k) @ U_bloch(k, params)
                        @ gamma_bloch(k).conj().T
                        - U_bloch(k, params).conj().T)) for k in ks)
report("Gamma U Gamma^-1 = U^dag (Bloch, all k)", err < 1e-12, f"err={err:.2e}")
G = gamma_real(Ns)
err = np.max(np.abs(G @ U @ G.conj().T - U.conj().T))
report("Gamma U Gamma^-1 = U^dag (real space)", err < 1e-12, f"err={err:.2e}")

print("E1d -- periodicity and projectivity")
err = max(np.max(np.abs(U_bloch(k + 2 * PI, params) - U_bloch(k, params)))
          for k in ks)
report("U(k+2pi) = U(k)", err < 1e-12, f"err={err:.2e}")
err = max(np.max(np.abs(gamma_bloch(k) @ gamma_bloch(k)
                        - np.exp(-1j * k) * np.eye(4))) for k in ks)
report("Gamma(k)^2 = e^{-ik} I", err < 1e-12, f"err={err:.2e}")

print("E1e -- twisted-basis block structure of H_eff")
worstleak = max(offdiag_det(k, params)[1] for k in np.linspace(0.1, 2 * PI, 23))
report("diagonal blocks of H_eff vanish", worstleak < 1e-9,
       f"leak={worstleak:.2e}")

print("E1f -- no k-independent chiral operator (no-collapse 1)")
# Solve X U(k) - U(k)^dag X = 0 for constant 4x4 X across sampled k.
rows = []
for k in np.linspace(0.05, 2 * PI, 9):
    Uk = U_bloch(k, params)
    rows.append(np.kron(np.eye(4), Uk.T) - np.kron(Uk.conj().T, np.eye(4)))
    # vec convention: vec(X U) = (U^T (x) I) vec(X); vec(U^dag X) = (I (x) U^dag) vec(X)
M = np.vstack([np.kron(U_bloch(k, params).T, np.eye(4))
               - np.kron(np.eye(4), U_bloch(k, params).conj().T)
               for k in np.linspace(0.05, 2 * PI, 9)])
nullity = 16 - np.linalg.matrix_rank(M, tol=1e-8)
report("constant-chiral solution space is empty", nullity == 0, f"dim={nullity}")

print("E1g -- trivial commutant of {U(k)} (no-collapse 2)")
M = np.vstack([np.kron(U_bloch(k, params).T, np.eye(4))
               - np.kron(np.eye(4), U_bloch(k, params))
               for k in np.linspace(0.05, 2 * PI, 9)])
nullity = 16 - np.linalg.matrix_rank(M, tol=1e-8)
report("commutant = scalars only", nullity == 1, f"dim={nullity}")

print("E1h -- Moebius transition of the twisted basis")
X = np.zeros((4, 4))
X[0, 3] = X[1, 2] = X[2, 1] = X[3, 0] = 1.0
k0 = 0.7
V1, V2 = twisted_basis(k0), twisted_basis(k0 + 2 * PI)
# find permutation/phase matrix P = V1^dag V2 and check it is the sector swap
P = V1.conj().T @ V2
offblock = np.linalg.norm(P[:2, :2]) + np.linalg.norm(P[2:, 2:])
report("V(k+2pi) = V(k) x (sector swap)", offblock < 1e-12,
       f"intra-sector residue={offblock:.2e}")
d1, _ = offdiag_det(k0, params)
d2, _ = offdiag_det(k0 + 2 * PI, params)
rel = abs(d2 - np.conj(d1)) / max(abs(d1), 1e-15)
report("d(k+2pi) == conj(d(k))", rel < 1e-9,
       f"d(k)={d1:.4f}, d(k+2pi)={d2:.4f}, rel={rel:.1e}")


# ---------------------------------------------------------------------------
# Additions of the 2026-11-12 revision.
# ---------------------------------------------------------------------------
from spaceglide_walker import (batched_F, U_from_F, batched_V, halfstep_blocks,
                               su2_coin, t_site, sx, I2, Pup, Pdn, rot)

print("E1f/g at 40 random real parameter points (the paper says 'generic')")
rng_fg = np.random.default_rng(31337)
bad_fg = 0
for _ in range(40):
    pr = tuple(rng_fg.uniform(-PI, PI, 4))
    Us = [U_bloch(k, pr) for k in np.linspace(0.05, 2 * PI, 9)]
    Mc = np.vstack([np.kron(Uk.T, np.eye(4)) - np.kron(np.eye(4), Uk.conj().T)
                    for Uk in Us])
    Mm = np.vstack([np.kron(Uk.T, np.eye(4)) - np.kron(np.eye(4), Uk)
                    for Uk in Us])
    bad_fg += int(16 - np.linalg.matrix_rank(Mc, tol=1e-8) != 0
                  or 16 - np.linalg.matrix_rank(Mm, tol=1e-8) != 1)
report("no constant chiral operator and scalar commutant at every point",
       bad_fg == 0, f"violations {bad_fg}/40")

print("E1i -- parity lemma: FULL spin-dependent shifts collapse the walker")
# F_full = Rc(beta) S Rc(alpha) with S moving up one site right, down one left.
Uks = []
for k in np.linspace(0.05, 2 * PI, 9):
    t = t_site(k)
    S = np.kron(t, Pup) + np.kron(t.conj().T, Pdn)
    Ca = np.zeros((4, 4), dtype=complex)
    Ca[:2, :2], Ca[2:, 2:] = rot(params[0]), rot(params[1])
    Cb = np.zeros((4, 4), dtype=complex)
    Cb[:2, :2], Cb[2:, 2:] = rot(params[2]), rot(params[3])
    Ffull = Cb @ S @ Ca
    G = np.kron(t, sx)
    Uks.append(G @ Ffull.conj().T @ G.conj().T @ Ffull)
Zsub = np.diag([1.0, 1.0, -1.0, -1.0])           # sublattice grading
err = max(np.abs(Zsub @ Uk - Uk @ Zsub).max() for Uk in Uks)
report("full-shift walker conserves the sublattice grading", err < 1e-12,
       f"||[Z, U(k)]|| = {err:.1e}")
M = np.vstack([np.kron(Uk.T, np.eye(4)) - np.kron(np.eye(4), Uk) for Uk in Uks])
nullity = 16 - np.linalg.matrix_rank(M, tol=1e-8)
report("its commutant is non-trivial (collapse)", nullity > 1, f"dim={nullity}")

print("E1j -- half-step block identities in the twisted frame")
rng = np.random.default_rng(12)
worst = dict(sv=0.0, mob=0.0, ublk=0.0, jac=0.0, detF=0.0)
for trial in range(40):
    coins = (params if trial == 0 else
             tuple(su2_coin(rng.normal(size=4)) for _ in range(4)))
    kk = rng.uniform(0, 2 * PI, 6)
    a, b, c, d = halfstep_blocks(kk, coins)
    a2, b2, c2, d2 = halfstep_blocks(kk + 2 * PI, coins)
    F, G = batched_F(kk, coins)
    U = U_from_F(F, G)
    V = batched_V(kk)
    Ut = np.conj(np.transpose(V, (0, 2, 1))) @ U @ V
    eps = np.sort(np.abs(np.angle(np.linalg.eigvals(U))), axis=1)[:, ::2]
    sb = np.sort(np.linalg.svd(b, compute_uv=False), axis=1)
    sd = np.sort(np.linalg.svd(d, compute_uv=False), axis=1)
    worst["sv"] = max(worst["sv"],
                      np.abs(sb - np.sort(np.sin(eps / 2), axis=1)).max(),
                      np.abs(sd - np.sort(np.cos(eps / 2), axis=1)).max())
    db, dd = np.linalg.det(b), np.linalg.det(d)
    worst["mob"] = max(worst["mob"],
                       np.abs(np.linalg.det(b2) - np.conj(db)).max(),
                       np.abs(np.linalg.det(d2) - np.conj(dd)).max())
    worst["jac"] = max(worst["jac"],
                       np.abs(np.linalg.det(a) - np.conj(dd)).max(),
                       np.abs(np.linalg.det(c) - np.conj(db)).max())
    aH = np.conj(np.transpose(a, (0, 2, 1)))
    worst["ublk"] = max(worst["ublk"],
                        np.abs(Ut[:, :2, 2:] - 2 * aH @ b).max(),
                        np.abs(Ut[:, :2, :2] - (2 * aH @ a - np.eye(2))).max(),
                        np.abs(np.linalg.det(Ut[:, :2, 2:]) - 4 * db * dd).max())
    worst["detF"] = max(worst["detF"], np.abs(np.linalg.det(F) - 1).max())
report("det F = 1", worst["detF"] < 1e-12, f"err={worst['detF']:.1e}")
report("sigma(b) = sin(eps/2), sigma(d) = cos(eps/2)", worst["sv"] < 1e-12,
       f"err={worst['sv']:.1e}")
report("det a = conj det d, det c = conj det b", worst["jac"] < 1e-12,
       f"err={worst['jac']:.1e}")
report("Moebius constraints on det b, det d", worst["mob"] < 1e-12,
       f"err={worst['mob']:.1e}")
report("U_++ = 2 a^dag a - 1, U_+- = 2 a^dag b, det U_+- = 4 det b det d",
       worst["ublk"] < 1e-12, f"err={worst['ublk']:.1e}")

print("E1k -- antiunitary symmetries of the real-coin walk")
k0 = 0.83
Uk, Umk = U_bloch(k0, params), U_bloch(-k0, params)
report("K: conj U(k) = U(-k)", np.abs(Uk.conj() - Umk).max() < 1e-12,
       f"err={np.abs(Uk.conj() - Umk).max():.1e}")
Tm = gamma_bloch(-k0)                    # T psi_k = Gamma(-k) conj(psi_k)
lhs = Tm @ Uk.conj() @ np.linalg.inv(Tm)
report("T = Gamma K: T U(k) T^-1 = U(-k)^dag",
       np.abs(lhs - Umk.conj().T).max() < 1e-12,
       f"err={np.abs(lhs - Umk.conj().T).max():.1e}")
T2 = gamma_bloch(k0) @ gamma_bloch(-k0).conj()
report("T^2 = e^{-ik}", np.abs(T2 - np.exp(-1j * k0) * np.eye(4)).max() < 1e-12,
       f"err={np.abs(T2 - np.exp(-1j * k0) * np.eye(4)).max():.1e}")
e = np.sort(-np.angle(np.linalg.eigvals(U_bloch(PI, params))))
report("Kramers pairs at k = pi", max(abs(e[0] - e[1]), abs(e[2] - e[3])) < 1e-12,
       f"eps(pi)/pi = {np.round(e / PI, 4)}")

print("\n" + ("E1 ALL PASS" if all(results) else ">>> E1 FAILURES <<<"))
sys.exit(0 if all(results) else 1)
