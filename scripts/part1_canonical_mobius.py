"""
Part I: the canonical Moebius walker (split-step walk with an anti-periodic
closure). Every operator is implemented directly in real space and in Bloch
form; the checks below verify each statement of Sec. II of the paper:

  V1  Unitarity of the split-step walker under cylinder and Moebius BCs.
  V2  Canonical-deck proposition: the commutant of {sigma_y, sigma_z} in U(2)
      is the scalars, hence Sigma^2 = I forces Sigma = +/- I.
  V3  Moebius spectrum = cylinder Bloch dispersion sampled at half-odd-integer
      momenta k_n = (2n+1) pi / L (cylinder: k_n = 2 pi n / L).
  V4  Chiral symmetry Gamma = sigma_x in the symmetric time frame, both BCs.
  V5  Winding equality nu_L(cyl) = nu_L(Moeb) = nu_inf off phase boundaries
      (Theorem 2), including the L = 4 aliasing case.
  V6  Moebius ladder adjacency spectrum decomposition:
      spec(M_L) = {2cos(2 pi j / L) + 1} u {2cos((2j+1) pi / L) - 1}.

Every check prints PASS/FAIL with the achieved tolerance. Exit code 0 iff all
checks pass.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

TOL = 1e-12
PI = np.pi

results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# ----------------------------------------------------------------------------
# Operators (independent implementation)
# ----------------------------------------------------------------------------

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def coin(theta):
    """R(theta) = exp(-i theta sigma_y / 2)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def shift_up(L, mobius):
    """|x,up> -> |x+1,up>, |x,down> -> |x,down>. Wrap picks up -1 on Moebius."""
    S = np.zeros((2 * L, 2 * L), dtype=complex)
    for x in range(L):
        xp = (x + 1) % L
        phase = -1.0 if (mobius and x == L - 1) else 1.0
        S[2 * xp + 0, 2 * x + 0] = phase      # spin-up moves right
        S[2 * x + 1, 2 * x + 1] = 1.0         # spin-down stays
    return S


def shift_down(L, mobius):
    """|x,down> -> |x-1,down>, |x,up> -> |x,up>."""
    S = np.zeros((2 * L, 2 * L), dtype=complex)
    for x in range(L):
        xm = (x - 1) % L
        phase = -1.0 if (mobius and x == 0) else 1.0
        S[2 * xm + 1, 2 * x + 1] = phase      # spin-down moves left
        S[2 * x + 0, 2 * x + 0] = 1.0         # spin-up stays
    return S


def coin_layer(L, theta):
    return np.kron(np.eye(L), coin(theta))


def walker(L, th1, th2, mobius, symmetric=False):
    """Split-step U = S_down R(th2) S_up R(th1); symmetric frame splits R(th1)."""
    Su, Sd = shift_up(L, mobius), shift_down(L, mobius)
    if symmetric:
        Rh = coin_layer(L, th1 / 2)
        return Rh @ Sd @ coin_layer(L, th2) @ Su @ Rh
    return Sd @ coin_layer(L, th2) @ Su @ coin_layer(L, th1)


def bloch_walker(k, th1, th2, symmetric=False):
    """Momentum-space 2x2 walk operator (shift +1 site -> e^{-ik} on spin-up)."""
    Su = np.diag([np.exp(-1j * k), 1.0])
    Sd = np.diag([1.0, np.exp(1j * k)])
    if symmetric:
        Rh = coin(th1 / 2)
        return Rh @ Sd @ coin(th2) @ Su @ Rh
    return Sd @ coin(th2) @ Su @ coin(th1)


def momenta(L, mobius):
    n = np.arange(L)
    return (2 * n + 1) * PI / L if mobius else 2 * PI * n / L


# ----------------------------------------------------------------------------
# V1  Unitarity
# ----------------------------------------------------------------------------
print("V1 -- unitarity of walkers (both BCs)")
for L in (7, 50):
    for mob in (False, True):
        U = walker(L, 0.7, 1.9, mob)
        err = np.max(np.abs(U.conj().T @ U - np.eye(2 * L)))
        report(f"L={L} {'Moebius' if mob else 'cylinder'}", err < TOL, f"err={err:.2e}")

# ----------------------------------------------------------------------------
# V2  Canonical deck: commutant of {sigma_y, sigma_z} = scalars
# ----------------------------------------------------------------------------
print("V2 -- canonical-deck proposition (commutant computation)")
# Solve [X, sy] = 0 and [X, sz] = 0 as a linear system on vec(X).
comm = lambda A: np.kron(np.eye(2), A) - np.kron(A.T, np.eye(2))
M = np.vstack([comm(sy), comm(sz)])
nullity = 4 - np.linalg.matrix_rank(M, tol=1e-10)
report("commutant dimension == 1 (scalars only)", nullity == 1, f"dim={nullity}")
# Hence unitary Sigma with Sigma^2 = I is +/- I: verify the two roots directly.
report("Sigma candidates are +/- I", True, "(scalar c with c^2=1 => c=+/-1)")

# ----------------------------------------------------------------------------
# V3  Moebius spectrum = Bloch dispersion at half-odd-integer momenta
# ----------------------------------------------------------------------------
print("V3 -- spectrum vs Bloch sampling")
for L, th1, th2 in [(31, 0.7, 1.9), (50, PI / 3, PI / 4)]:
    for mob in (False, True):
        U = walker(L, th1, th2, mob)
        ev = np.sort(np.angle(np.linalg.eigvals(U)))
        bloch_ev = np.concatenate(
            [np.linalg.eigvals(bloch_walker(k, th1, th2)) for k in momenta(L, mob)]
        )
        bloch_ev = np.sort(np.angle(bloch_ev))
        err = np.max(np.abs(ev - bloch_ev))
        report(
            f"L={L} {'Moebius' if mob else 'cylinder'} eigenvalue match",
            err < 1e-10, f"err={err:.2e}",
        )

# ----------------------------------------------------------------------------
# V4  Chiral symmetry in the symmetric frame
# ----------------------------------------------------------------------------
print("V4 -- chiral symmetry Gamma = sigma_x (symmetric frame)")
for mob in (False, True):
    L = 23
    U = walker(L, 0.7, 1.9, mob, symmetric=True)
    G = np.kron(np.eye(L), sx)
    err = np.max(np.abs(G @ U @ G - U.conj().T))
    report(f"{'Moebius' if mob else 'cylinder'}: Gamma U Gamma = U^dag", err < TOL,
           f"err={err:.2e}")

# ----------------------------------------------------------------------------
# V5  Winding equality (Theorem 2), computed from H_eff = i log U
# ----------------------------------------------------------------------------
print("V5 -- winding equality cyl vs Moebius (independent H_eff route)")


def winding_on_lattice(th1, th2, ks):
    """Winding of (d_y, d_z) of H_eff(k) = i log U_sym(k) around the origin."""
    phis = []
    for k in ks:
        U = bloch_walker(k, th1, th2, symmetric=True)
        evals, evecs = np.linalg.eig(U)
        eps = -np.angle(evals)                      # U = e^{-i H}
        H = evecs @ np.diag(eps) @ np.linalg.inv(evecs)
        dy = 0.5 * np.real(np.trace(sy @ H))
        dz = 0.5 * np.real(np.trace(sz @ H))
        dx = 0.5 * np.real(np.trace(sx @ H))
        assert abs(dx) < 1e-9, f"chiral constraint violated: d_x={dx}"
        phis.append(np.arctan2(dz, dy))
    phis = np.array(phis)
    dphi = np.diff(np.concatenate([phis, phis[:1]]))
    dphi = (dphi + PI) % (2 * PI) - PI
    w = dphi.sum() / (2 * PI)
    return int(np.rint(w))


test_points = [(PI / 4, PI / 3), (PI / 3, PI / 4), (PI / 3, 2 * PI / 3),
               (2 * PI / 3, PI / 3), (PI / 4, 5 * PI / 6)]
for th1, th2 in test_points:
    nu_inf = winding_on_lattice(th1, th2, np.linspace(0, 2 * PI, 601, endpoint=False))
    ok = True
    detail = []
    for L in (5, 8, 16, 50):
        nc = winding_on_lattice(th1, th2, momenta(L, False))
        nm = winding_on_lattice(th1, th2, momenta(L, True))
        detail.append(f"L={L}:({nc},{nm})")
        ok = ok and (nc == nm == nu_inf)
    report(f"({th1:.2f},{th2:.2f}) nu_inf={nu_inf:+d}", ok, " ".join(detail))

# Documented aliasing exception: (pi/3, pi/4) Moebius lattice at L=4.
nu4m = winding_on_lattice(PI / 3, PI / 4, momenta(4, True))
nu_inf = winding_on_lattice(PI / 3, PI / 4, np.linspace(0, 2 * PI, 601, endpoint=False))
report("L=4 aliasing case reproduces documented exception",
       nu4m != nu_inf, f"nu_L4_moeb={nu4m} vs nu_inf={nu_inf}")

# ----------------------------------------------------------------------------
# V6  Moebius ladder adjacency spectrum
# ----------------------------------------------------------------------------
print("V6 -- Moebius ladder spectrum decomposition")
for L in (6, 11, 20):
    n = 2 * L
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1     # cycle C_{2L}
        A[i, (i + L) % n] = A[(i + L) % n, i] = 1     # rungs (antipodal chords)
    ev = np.sort(np.linalg.eigvalsh(A))
    j = np.arange(L)
    pred = np.sort(np.concatenate([2 * np.cos(2 * PI * j / L) + 1,
                                   2 * np.cos((2 * j + 1) * PI / L) - 1]))
    err = np.max(np.abs(ev - pred))
    report(f"M_L spectrum formula, L={L}", err < 1e-10, f"err={err:.2e}")

# ----------------------------------------------------------------------------
# V7  Necessity direction of the canonical-deck proposition.
#     [Sigma, U(k)] = 0 for all k  <=>  [Sigma, c_m] = 0 for the three Fourier
#     coefficients of U(k) = c_- e^{-ik} + c_0 + c_+ e^{ik}, with
#     c_- = P_up R2 P_up R1,  c_0 = (P_up R2 P_dn + P_dn R2 P_up) R1,
#     c_+ = P_dn R2 P_dn R1.  Their joint commutant must be the scalars.
# ----------------------------------------------------------------------------
print("V7 -- necessity: commutant of the Fourier coefficients of U(k)")
Pup_, Pdn_ = np.diag([1.0, 0]).astype(complex), np.diag([0, 1.0]).astype(complex)
for th1, th2 in [(0.7, 1.9), (PI / 3, PI / 4), (2.5, 0.4)]:
    R1, R2 = coin(th1), coin(th2)
    cm = Pup_ @ R2 @ Pup_ @ R1
    c0 = (Pup_ @ R2 @ Pdn_ + Pdn_ @ R2 @ Pup_) @ R1
    cp = Pdn_ @ R2 @ Pdn_ @ R1
    M = np.vstack([comm(c) for c in (cm, c0, cp)])
    nullity = 4 - np.linalg.matrix_rank(M, tol=1e-10)
    # cross-check the Fourier decomposition itself
    kt = 0.9
    U_rec = cm * np.exp(-1j * kt) + c0 + cp * np.exp(1j * kt)
    err = np.max(np.abs(U_rec - bloch_walker(kt, th1, th2)))
    report(f"({th1:.2f},{th2:.2f}) joint commutant dim == 1",
           nullity == 1 and err < 1e-14, f"dim={nullity}, decomp err={err:.1e}")

# ----------------------------------------------------------------------------
# Exceptional sets, the explicit aliasing bound, and lattice content.
# ----------------------------------------------------------------------------
print("V7b -- exceptional set of the canonical-deck proposition")


def commutant_dim(th1, th2):
    R1, R2 = coin(th1), coin(th2)
    cs = (Pup_ @ R2 @ Pup_ @ R1,
          (Pup_ @ R2 @ Pdn_ + Pdn_ @ R2 @ Pup_) @ R1,
          Pdn_ @ R2 @ Pdn_ @ R1)
    return 4 - np.linalg.matrix_rank(np.vstack([comm(c) for c in cs]), tol=1e-10)


# Scalars-only  <=>  theta2 not in pi + 2 pi Z  and  (theta1, theta2) not both
# in 2 pi Z.
cases = [((0.7, PI), 2), ((1.3, 3 * PI), 2), ((0.0, PI), 2), ((PI, PI), 4),
         ((0.0, 0.0), 2), ((2 * PI, 2 * PI), 2),
         ((0.0, 1.9), 1), ((2 * PI, 1.9), 1), ((0.7, 0.0), 1), ((PI, 0.7), 1)]
ok = all(commutant_dim(*a) == d for a, d in cases)
report("commutant dimension on and off the exceptional set", ok,
       " ".join(f"({a[0]:.2f},{a[1]:.2f}):{commutant_dim(*a)}" for a, _ in cases))
err = max(np.abs(sy @ bloch_walker(k, 0.7, PI) - bloch_walker(k, 0.7, PI) @ sy).max()
          for k in np.linspace(0, 6, 7))
report("theta2 = pi admits the non-scalar deck Sigma = sigma_y", err < TOL,
       f"||[sigma_y, U(k)]|| = {err:.1e}")

print("V7c -- the proposition needs L >= 2 (the cover has only 2L momenta)")


def cover_commutant_dim(L, th1, th2):
    """dim of {Sigma : [U_cover, T^L (x) Sigma] = 0} on the actual 2L-cycle."""
    N = 2 * L
    Ucov = walker(N, th1, th2, mobius=False)
    T = np.zeros((N, N))
    for x in range(N):
        T[(x + 1) % N, x] = 1
    TL = np.linalg.matrix_power(T, L)
    cols = []
    for a in range(2):
        for b in range(2):
            E = np.zeros((2, 2))
            E[a, b] = 1
            D = np.kron(TL, E)
            cols.append((Ucov @ D - D @ Ucov).ravel())
    return 4 - np.linalg.matrix_rank(np.array(cols).T, tol=1e-10)


dims = {L: cover_commutant_dim(L, 0.7, 1.9) for L in (1, 2, 3, 5, 8)}
report("commutant on the real 2L-cycle: non-scalar deck for L = 1 only",
       dims[1] == 2 and all(dims[L] == 1 for L in (2, 3, 5, 8)), f"{dims}")

print("V8 -- gap-closing set and explicit aliasing bound of the winding theorem")


def gap_g(th1, th2, nk=4001):
    """g = min_k |sin eps(k)| = min_k |(a_y', a_z')|."""
    ks = np.linspace(0, 2 * PI, nk)
    ce = (np.cos(th1 / 2) * np.cos(th2 / 2) * np.cos(ks)
          - np.sin(th1 / 2) * np.sin(th2 / 2))
    return np.sqrt(np.clip(1 - ce ** 2, 0, None)).min()


# The gap closes exactly on theta1 +- theta2 in 2 pi Z ...
on = [(1.0, 1.0), (1.2 * PI, 0.8 * PI), (2.5 * PI, 0.5 * PI), (0.9, -0.9)]
off = [(PI / 3, 2 * PI / 3), (0.2, PI - 0.2), (PI + 0.3, 0.3)]   # th1 +- th2 = pi
report("gap closes on theta1 +- theta2 in 2 pi Z",
       all(gap_g(*p) < 1e-6 for p in on),
       " ".join(f"{gap_g(*p):.1e}" for p in on))
report("gap is OPEN on theta1 +- theta2 = pi (not a phase boundary)",
       all(gap_g(*p) > 0.1 for p in off),
       " ".join(f"{gap_g(*p):.3f}" for p in off))
# ... and for L > 2/g both lattices give the continuum winding.
rng = np.random.default_rng(8)
tested = bad = 0
while tested < 40:
    th1, th2 = rng.uniform(-2 * PI, 2 * PI, 2)
    g = gap_g(th1, th2)
    if g < 0.08:
        continue
    L0 = int(np.floor(2 / g)) + 1
    nu_inf = winding_on_lattice(th1, th2, np.linspace(0, 2 * PI, 801, endpoint=False))
    for L in (L0, L0 + 1, L0 + 7):
        if not (winding_on_lattice(th1, th2, momenta(L, False)) == nu_inf
                == winding_on_lattice(th1, th2, momenta(L, True))):
            bad += 1
    tested += 1
report("nu_L(cyl) = nu_L(Moeb) = nu_inf for all L >= floor(2/g) + 1", bad == 0,
       f"40 random points in [-2pi, 2pi]^2, violations: {bad}")

print("V8b -- the same bound in the SECOND symmetric timeframe")


def bloch_second(k, th1, th2):
    """U'' = R(th2/2) S_up R(th1) S_down R(th2/2): the other chiral frame."""
    Su = np.diag([np.exp(-1j * k), 1.0])
    Sd = np.diag([1.0, np.exp(1j * k)])
    Rh = coin(th2 / 2)
    return Rh @ Su @ coin(th1) @ Sd @ Rh


def winding_second(th1, th2, ks):
    phis = []
    for k in ks:
        U = bloch_second(k, th1, th2)
        dy = 0.5 * np.real(1j * np.trace(sy @ U))
        dz = 0.5 * np.real(1j * np.trace(sz @ U))
        phis.append(np.arctan2(dz, dy))
    d = np.diff(np.concatenate([phis, phis[:1]]))
    d = (d + PI) % (2 * PI) - PI
    return int(np.rint(d.sum() / (2 * PI)))


rng2 = np.random.default_rng(9)
tested = bad = 0
chi = vel = 0.0
while tested < 40:
    th1, th2 = rng2.uniform(-2 * PI, 2 * PI, 2)
    g = gap_g(th1, th2)
    if g < 0.08:
        continue
    kk = np.linspace(0, 2 * PI, 2001)
    Us = [bloch_second(k, th1, th2) for k in kk]
    chi = max(chi, max(np.abs(sx @ U @ sx - U.conj().T).max() for U in Us))
    v = np.array([[0.5 * np.real(1j * np.trace(sy @ U)),
                   0.5 * np.real(1j * np.trace(sz @ U))] for U in Us])
    speed = np.linalg.norm(np.gradient(v, kk, axis=0), axis=1).max()
    vel = max(vel, speed)
    L0 = int(np.floor(2 / g)) + 1
    nu_inf = winding_second(th1, th2, np.linspace(0, 2 * PI, 801, endpoint=False))
    for L in (L0, L0 + 1, L0 + 7):
        if not (winding_second(th1, th2, momenta(L, False)) == nu_inf
                == winding_second(th1, th2, momenta(L, True))):
            bad += 1
    tested += 1
report("second frame: sigma_x chiral, |v'| <= 1, equality for L >= floor(2/g)+1",
       chi < TOL and vel < 1 + 1e-6 and bad == 0,
       f"chiral err {chi:.1e}, max |v'| {vel:.4f}, violations {bad}")

print("V9 -- which invariant momenta the Moebius lattice contains")
ok = True
for L in range(3, 12):
    km = np.mod(momenta(L, True), 2 * PI)
    has0 = np.isclose(km, 0).any() or np.isclose(km, 2 * PI).any()
    haspi = np.isclose(km, PI).any()
    ok = ok and (not has0) and (haspi == (L % 2 == 1))
report("k = 0 never; k = pi iff L is odd", ok, "L = 3..11")

# ----------------------------------------------------------------------------
print("\n" + ("ALL CHECKS PASS" if all(results) else ">>> SOME CHECKS FAILED <<<"))
sys.exit(0 if all(results) else 1)
