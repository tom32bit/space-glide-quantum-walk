# Non-Orientable Discrete-Time Quantum Walks: A No-Go Theorem and Its Nonsymmorphic Resolution

**Authors:** S. M. Yousuf Iqbal Tomal, Abdullah Al Shafin
**Status:** manuscript v4 (2026-11-13) + complete verified numerical pipeline
**License:** MIT

One paper, two movements:

- **Part I (no-go).** For the chiral split-step DTQW, the only Möbius deck
  operators compatible with the walk are scalars (Σ = ±𝕀) outside an explicit
  exceptional set of coin angles (θ₂ ∈ π + 2πℤ, where Σ = σ_y is also allowed).
  The scalar twist is a ℤ₂ holonomy (a π flux): the spectrum is the cylinder
  dispersion at half-odd-integer momenta, and the winding numbers equal the
  cylinder's for every size L > 2/g, with g = min_k |sin ε(k)|. The twist
  leaves the chiral winding classification unchanged. (A quantum walk *can*
  detect non-orientability as a sign along paths — Higuchi & Segawa,
  arXiv:2501.06765; the statement here concerns the bulk classification only.)
- **Part II (resolution).** The glide-chiral operator Γ = (one-site
  shift)·σx obeys Γ(k)² = e^{-ik}𝕀; its eigenbundle is a **Möbius bundle over
  the Brillouin zone**. Every local half-step F yields a glide-chiral walk
  U = ΓF†Γ⁻¹F. In the twisted Γ-eigenbasis, F = [[a, b],[c, d]] has
  σ(b) = sin(ε/2), σ(d) = cos(ε/2), and det b, det d each obey a Möbius
  constraint: they define **two gap-resolved ℤ₂ invariants (ζ₀, ζ_π)**, each
  needing only its own gap and only the unitary glide symmetry. The Floquet
  operator alone fixes only ζ₀ + ζ_π. For real coins the phase diagram is
  known in closed form (all four classes) and a glide time reversal
  T = ΓK, T² = e^{-ik}, enforces Kramers pairs at k = π.
- **Defects (proven).** For real coins every domain wall binds
  N₀ ≡ Δζ₀ and N_π ≡ Δζ_π (mod 2) modes at quasi-energies 0 and π. The proof
  uses the complete gap-closing set (the invariants' generating functions are
  quadratics in sin(k/2)) and one exactly solvable wall. These modes are
  protected by the particle-hole symmetry K of real coins (class D), **not**
  by the glide: a local K-breaking perturbation removes them, and a local
  real operation of determinant −1 shifts both counts by one.

## Layout

```text
paper/                      manuscript (REVTeX) + bibliography  [not in git]
theory/part1_no_go/         theorem notes T-D1..T-D7 (working notes; errata on top)
theory/part2_nonsymmorphic/ construction theorem, parity lemma, invariants
code/part1_validation/      validation suite for Part I + manuscript checker
code/part2_spaceglide/      walker library + experiments E1..E12
code/figures/               generated figures + raw npz data
code/legacy_notebook/       original Option A notebook (archival, superseded)
docs/                       unified spec, validation log, lit checks, analyses
archive/                    former Option A paper, failed Option B candidates
```

## Reproduce everything

```bash
pip install -r requirements.txt
python code/part1_validation/validate_part1.py        # Part I: V1..V9, ~30 s
cd code/part2_spaceglide
python e1_construction_checks.py                      # E1: construction + block identities
python run_gap_resolved.py                            # E9: invariant audits + four-phase figure
python run_phase_diagram.py                           # E3: slice audit, ~3 min
python run_bands_figure.py                            # E2 figure
python run_walls_and_bridge.py                        # E4/E5/E6: walls, bridge, dynamics
python run_revision_defects.py                        # E7: what protects the wall modes
python run_revision_bulk.py                           # E8: branch-cut identity, second slice
python run_wall_theorem.py                            # E10: every step of the wall-rule proof
python run_robustness.py                              # E11: thresholds, sizes, boundaries, ~5 min
python run_richer_family.py                           # E12: four-layer family, ~10 min
```

Every figure in the paper regenerates from these scripts; every theorem cites
its verification block. `docs/VALIDATION_LOG.md` records the audits, including
the 2026-11-12 round that corrected the invariant and defect claims of the
first draft.

## Provenance

This repository unifies and supersedes two earlier project lines ("Option A":
canonical Möbius DTQW characterisation; "Option B": intrinsic Möbius walker).
Their archived states are under `archive/` and `docs/archive/`.
