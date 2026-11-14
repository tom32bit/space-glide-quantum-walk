# Unified Research Specification

**Paper:** *Non-Orientable Discrete-Time Quantum Walks: A No-Go Theorem and Its
Nonsymmorphic Resolution*
**Authors:** S. M. Yousuf Iqbal Tomal, Abdullah Al Shafin
**Spec version:** 3.0 (2026-11-12). Supersedes v2.0 (2026-07-19) and the
`archive/` specs of both former options. v3.0 records the corrections of the
2026-11-12 audit (see `docs/VALIDATION_LOG.md`, revision round 2): the
invariant is now the gap-resolved pair (ζ₀, ζ_π) from the half-step, and the
wall modes are attributed to the particle-hole symmetry K, not to the glide.

---

## 1. Thesis (one paragraph)

A Möbius twist put into the boundary condition of the chiral split-step DTQW
is forced to be scalar (outside an explicit exceptional set of coin angles)
and leaves the winding classification unchanged at every size L > 2/g
(**Part I**, proven + machine-verified). Put into the *symmetry algebra* it is
active: the **glide-chiral** operator Γ = (one-site translation)·σx with
Γ(k)² = e^{-ik}𝕀 has a **Möbius bundle** of eigenspaces over the Brillouin
zone, every half-step F gives a glide-chiral walk U = ΓF†Γ⁻¹F, and the blocks
b, d of F in the twisted eigenbasis (singular values sin(ε/2), cos(ε/2)) carry
two **gap-resolved ℤ₂ invariants (ζ₀, ζ_π)** protected by the unitary glide
alone; U by itself fixes only ζ₀ + ζ_π (**Part II**, proven +
machine-verified). For real coins the phase diagram is closed-form and a glide
time reversal gives Kramers pairs at k = π. Domain walls obey
N_μ ≡ Δζ_μ (mod 2), but those modes are protected by the particle-hole
symmetry K (class D), not by the glide, which has no boundary signature in 1D.

## 2. Deliverables and status

| ID | Deliverable | Status |
|---|---|---|
| D1 | Part I re-validation suite (independent code) | DONE — `code/part1_validation/validate_part1.py`, all PASS |
| D2 | Part I theory (T-D1…T-D7, inherited, re-verified) | DONE — `theory/part1_no_go/` |
| D3 | Part II construction theorem + parity lemma + invariant | DONE — `theory/part2_nonsymmorphic/P2_construction_and_invariant.md` |
| D4 | Part II library + E1 checks | DONE — `code/part2_spaceglide/` |
| D5 | E2 bands/winding figure; E3 phase diagram; E4 walls; E5 bridge; E6 dynamics | DONE — `code/figures/E*.png` + npz |
| D6 | Refreshed lit check | DONE — `docs/lit_check_2026-07-19.md` |
| D7 | Unified manuscript | REVISED v3 (2026-11-12) — `paper/main.tex`; mechanical check + command lint pass; **never compiled** (no TeX locally) |
| D8 | Internal review + revision round | DONE — E7/E8 experiments, `check_manuscript.py`, logged in `docs/VALIDATION_LOG.md` |
| D9 | Open-problem section backlog (below) | ONGOING |
| D10 | Adversarial audit + correction round | DONE 2026-11-12 — 7 errors fixed, bibliography re-verified, E9 + E1i–k + E7d + V7b–V9 added |
| D11 | Proof of the wall rule + robustness study | DONE 2026-11-13 — manuscript v4: closed form at all k, Theorem 5 (wall rule), Proposition 3 (surgery); E10 (proof steps), E11 (robustness), E12 (four-layer family) |

## 3. Definitions locked (conventions)

- Split-step: U = S↓ R(θ₂) S↑ R(θ₁); rotations R(θ) = e^{-iθσy/2};
  symmetric frame splits R(θ₁). Winding orientation as in
  `validate_part1.py` (opposite sign to the 2026-05 docs — see VALIDATION_LOG).
- Part II cell: (A↑, A↓, B↑, B↓); Bloch gauge |k,s,c⟩ ∝ Σₙ e^{ikn}|2n+s, c⟩.
- Glide Γ(k) = t(k)⊗σx, t(k) = [[0, e^{-ik}],[1,0]].
- Half-protocol F(k) = S↓(k) Rc(β) S↑(k) Rc(α), partial shifts, sublattice
  coin angles α=(αA,αB), β=(βA,βB).
- Invariants: (ζ₀, ζ_π) = half-zone parities of det b, det d (P2 §8), canonical
  frame, real F. Legacy H_eff parity = ζ₀ + ζ_π + 1 (cross-check only).
- Paper figure: plane (αA+αB, βA+βB) at αA−αB = −0.628, βA−βB = 0.9.
  Audit slices: (αA, αB) ∈ [−π,π]² at (βA,βB) = (−1.24,−1.39) and (0.4, 1.3).
- Class representatives (ζ₀, ζ_π): A (0,0) = (2.0944, 2.7227, −1.24, −1.39);
  B (0,1) = (2.0944, 1.1519, −1.24, −1.39); C (1,1) = (2.8274, −1.0996,
  −1.24, −1.39); D (1,0) = (−0.6355, −2.4256, −3.0049, −3.1354).

## 4. Claims discipline

Claimed as theorems (proofs in the manuscript): Proposition 1 with its exact
exceptional set; Theorems 1–2 (sampling; winding equality for L > 2/g);
construction theorem; parity lemma; half-step block lemma; gap-resolved ℤ₂
theorem with its conventions; closed form for real coins at all momenta and
the complete gap-closing set; the wall rule N_μ ≡ Δζ_μ for real-coin walls of
the two-layer family (Theorem 5); the local-surgery rule (Proposition 3).
Claimed as machine-verified numerics: K-pinning statistics, hybridization
rates against the analytic values, detection amplitudes, and everything for
the four-layer family. Explicitly NOT claimed: that the glide protects wall
modes (it does not); the wall rule beyond the two-layer family as a theorem;
classification completeness; a finer invariant at the Kramers closings (only
suggested by E12b); any boundary signature without K.

## 5. Roadmap after the draft

1. Author review of the manuscript; fix affiliation/email placeholders.
2. Re-run `docs/lit_check` queries; then arXiv (quant-ph, cross-list
   cond-mat.mes-hall).
3. Target venue: Physical Review B or Physical Review Research (PRR preferred:
   construction + numerics + open theory items fit its format).
4. Follow-up project (separate): projective-symmetry-protected gate protection
   using this walker as the platform (the 2026-05-16 "Option B of the deep
   analysis"); the S_π detector and wall modes are the starting resources.
5. Optional strengthening if a referee asks: entanglement-spectrum probe of
   the ℤ₂; richer F families to decouple ζ₀ from ζπ; K-theory classification.

## 6. Reproduction

- Part I: `python code/part1_validation/validate_part1.py` (V1–V9, ~30 s).
- Part II checks: `e1_construction_checks.py` (E1a–k), `run_gap_resolved.py`
  (E9 audits + Fig. 2), `run_phase_diagram.py` (E3 slice audit, ~3 min),
  `run_revision_bulk.py` (E8).
- Figures 1, 3, 4: `run_bands_figure.py`, `run_walls_and_bridge.py`,
  `run_revision_defects.py`.
Requirements: numpy, scipy, matplotlib (see `requirements.txt`).
