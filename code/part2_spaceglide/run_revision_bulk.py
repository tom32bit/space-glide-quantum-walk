"""
E8 -- Bulk cross-checks.

E8a  Reality-broken family (sublattice coin phases, K broken): the H_eff-based
     half-zone parity stays quantized, and its two branch-cut versions agree
     at every gapped point. Revision 2026-11-12: this agreement is an exact
     identity (H' - pi = H - pi sgn H, so the two flattened Hamiltonians are
     negatives of each other and det(-D) = det D for 2x2 blocks); the first
     draft listed it as an open observation. The identity is checked here at
     operator level as well.

E8b  Second beta slice (bA, bB) = (0.4, 1.3), 41x41: the gap-resolved audit of
     E3 repeated on an independent slice (quantization, closed form,
     gap-selective boundary alignment).
"""

import numpy as np
from spaceglide_walker import (PI, point_data, batched_U, gap_resolved_z2,
                               closed_form_z2)

# ------------------------------------------------------------------- E8a

print("E8a -- reality-broken family: H_eff parity, both branch cuts")
rng = np.random.default_rng(2026)
classes = {}
qmax = 0.0
ident = 0.0
gapped = 0
NPTS = 60
for _ in range(NPTS):
    p = tuple(rng.uniform(-PI, PI, 4))
    ph = tuple(rng.uniform(-PI, PI, 2))
    g0, gp, (z_cut_pi, _, _, q0), (z_cut_0, _, _, qp) = point_data(
        p, nk=257, phases=ph)
    if g0 > 0.05 and gp > 0.05:
        gapped += 1
        qmax = max(qmax, q0, qp)
        classes[(z_cut_pi, z_cut_0)] = classes.get((z_cut_pi, z_cut_0), 0) + 1
        # operator-level identity at a few momenta
        U = batched_U(np.array([0.3, 2.2, 4.4]), p, ph)
        ev, V = np.linalg.eig(U)
        eps = -np.angle(ev)
        Vi = np.linalg.inv(V)
        H = np.einsum("nij,nj,njk->nik", V, eps, Vi)
        e2 = np.where(eps <= 0, eps + 2 * PI, eps)
        H2 = np.einsum("nij,nj,njk->nik", V, e2 - PI, Vi)
        sg = np.einsum("nij,nj,njk->nik", V, np.sign(eps), Vi)
        ident = max(ident, np.abs(H2 - (H - PI * sg)).max())
print(f"  gapped points: {gapped}/{NPTS}")
print(f"  max quantization error on gapped points: {qmax:.2e}")
print(f"  (parity cut pi, parity cut 0) counts: {classes}")
print(f"  both cuts agree everywhere: "
      f"{all(a == b for a, b in classes)}")
print(f"  identity H' - pi = H - pi sgn(H): max deviation {ident:.1e}")

# ------------------------------------------------------------------- E8b

print("\nE8b -- second beta slice (bA, bB) = (0.4, 1.3), 41x41, gap-resolved")
BFIX = (0.4, 1.3)
NG = 41
grid = np.linspace(-PI, PI, NG)
Z0 = np.zeros((NG, NG), dtype=int)
ZP = np.zeros((NG, NG), dtype=int)
G0 = np.zeros((NG, NG))
GP = np.zeros((NG, NG))
QE = np.zeros((NG, NG))
CF = np.zeros((NG, NG), dtype=bool)
for i, aA in enumerate(grid):
    for j, aB in enumerate(grid):
        p = (aA, aB) + BFIX
        r = gap_resolved_z2(p, nk=161)
        Z0[i, j], ZP[i, j] = r["zeta0"], r["zetapi"]
        G0[i, j], GP[i, j], QE[i, j] = r["gap0"], r["gappi"], r["qerr"]
        CF[i, j] = (r["zeta0"], r["zetapi"]) == closed_form_z2(p)[:2]

gapped = np.minimum(G0, GP) > 0.05
print(f"  gapped points: {gapped.sum()} of {NG * NG}")
print(f"  quantization error on gapped points: {QE[gapped].max():.2e}")
print(f"  closed form agrees on gapped points: {CF[gapped].sum()} of "
      f"{gapped.sum()}")
print(f"  classes present: "
      f"{sorted(set(zip(Z0[gapped].tolist(), ZP[gapped].tolist())))}")
for name, Z, G in (("zeta_0", Z0, G0), ("zeta_pi", ZP, GP)):
    njump = bad = 0
    worst = 0.0
    for ax in (0, 1):
        dz = np.diff(Z, axis=ax) != 0
        own = np.minimum(np.take(G, range(0, NG - 1), axis=ax),
                         np.take(G, range(1, NG), axis=ax))
        njump += dz.sum()
        bad += (dz & (own > 0.25)).sum()
        if dz.any():
            worst = max(worst, own[dz].max())
    print(f"  {name}: {njump} jumps; largest OWN-gap floor on a jump step "
          f"{worst:.3f}; jumps with own gap > 0.25: {bad}")
np.savez("../figures/E8_second_slice.npz", grid=grid, Z0=Z0, ZP=ZP,
         G0=G0, GP=GP, QE=QE, BFIX=np.array(BFIX))
print("  saved ../figures/E8_second_slice.npz")
