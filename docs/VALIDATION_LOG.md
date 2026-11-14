# Validation Log — Unified Paper Restart (2026-07-19)

Per the restart directive: **nothing inherited from the Option A / Option B phase
was taken as ground truth.** Every load-bearing claim was re-derived or
re-verified with independent code (`code/part1_validation/validate_part1.py`,
written from scratch — no code shared with the legacy notebook), and every new
Part II claim was built with adversarial checks (collapse tests) designed to
kill it. This file records what was questioned, what survived, what was
corrected, and what was discovered in the process.

## Part I claims (inherited) — all re-verified independently

| Claim | Re-check | Outcome |
|---|---|---|
| Unitarity of split-step walkers, both BCs | fresh implementation, L ∈ {7,50} | PASS, err ~2e-16 |
| Canonical-deck proposition (Σ ∈ {±𝕀}) | commutant of {σy, σz} computed as a linear system | PASS, nullity 1 (scalars) |
| Möbius spectrum = Bloch at half-odd-integer momenta | direct diagonalisation vs analytic Bloch, L ∈ {31,50} | PASS, err ≤ 3e-15 |
| Chiral symmetry Γ = σx in symmetric frame, both BCs | operator identity | PASS, err ≤ 6e-17 |
| T-D4 winding equality (cyl = Möb = ν∞) | independent H_eff route (no legacy formulas), 5 interior points × L ∈ {5,8,16,50} | PASS |
| T-D4 documented L=4 aliasing exception | reproduced | PASS |
| T-D7 Möbius ladder spectrum formula | direct graph diagonalisation, L ∈ {6,11,20} | PASS, err ≤ 7e-15 |

**Corrections / sharpenings found:**

1. **Winding sign convention.** The legacy docs' winding values are opposite in
   sign to the convention used here (their −1 = our +1). The *equality* claims
   are unaffected. The unified paper fixes one convention and states it.
2. **Canonical-deck proposition wording.** The old statement required Σ to
   commute with R(θ) "for all θ"; only fixed generic angles are ever used. The
   commutant of a single generic rotation already equals the commutant of σy,
   so the proposition holds pointwise for generic (θ₁, θ₂), with exceptions
   only at the measure-zero special angles. The unified paper states it this
   way (Part I, Proposition 1).

## Part II claims (new) — adversarial checks and live corrections

1. **First construction collapsed — caught by design.** The initial
   half-protocol used the *full* spin-dependent shift; the no-collapse tests
   (E1f/E1g) found a 4-dimensional commutant. Diagnosis: displacement-parity
   argument — U = ΓF†Γ⁻¹F with all-odd-parity F factors conserves the
   sublattice grading. Promoted to a design lemma (partial shifts are
   necessary); split-step F passes all tests (commutant = scalars, no constant
   chiral operator). *This is exactly the failure mode of the two 2026-05
   Option B candidates, now understood structurally.*
2. **First invariant proposal was wrong — caught by probe.** The naive
   MTI-style sign pair sgn d(0)·sgn d(2π) is identically +1 because
   d(2π) = conj(d(0)) is an algebraic identity, not independent data. The
   correct invariant is the relative half-zone winding parity ζ = n mod 2
   (P2 §5). Quantization then verified to ≤ 3e-15 over the full scanned grid.
3. **Protection scope tested.** ζ remains exactly quantized when coin phases
   break the reality symmetry — the invariant needs glide-chirality only
   (stronger claim, matching the "Z2 from unitary symmetries alone" narrative).
4. **Control-experiment artifact caught.** The first domain-wall control used
   the hyper-symmetric point (−π/2, −π/2) and showed spurious near-pinned
   quadruplets. A generic trivial control shows zero in-gap wall modes. The
   published control uses the generic point.
5. **Boundary-alignment audit.** Every ζ jump on the 61×61 phase-diagram grid
   sits on a step whose bulk-gap floor is ≤ 0.005; there are zero ζ jumps
   across well-gapped steps.

## Revision round (2026-07-20, internal review)

An internal review ran three adversarial numerical tests against the draft
and found two substantive flaws; the full 10-item revision list was then
executed:

1. **Theorem 4 overclaim [FIXED].** Absolute ζ shown frame-relative (an
   admissible gauge with det g± = ±e^{ik/2} shifts n by −1 at every
   parameter point). Theorem restated: canonical-frame ζ + walk-independent
   frame shifts + intrinsic Δζ, with proof of all three parts.
2. **Wall-pinning misattribution [FIXED].** Mechanism identified and
   demonstrated (E7a): reality/PHS K pins (real disorder, even global at
   W = 0.3, keeps deviation 2–5e-5; complex phase disorder unpins →1e-2);
   glide Δζ dictates existence. Defect section rewritten around this.
3. Prop 1 necessity proven (Fourier-coefficient commutant argument, V7:
   joint commutant dim 1 at three generic angle pairs, decomposition error
   ≤ 1e-16).
4. Inter-wall hybridization: ξ = 1.41 cells, r = −0.999994 over six
   decades; the s ≳ 30 upturn identified as the tanh-profile tail floor
   (~1e-8), NOT second-wall hybridization (would be ~1e-15) — corrected
   in figure and text before it could mislead.
5. Second β-slice (0.4, 1.3), 41×41: both classes, zero audit violations,
   quantization 1.7e-15 (E8b).
6. Reality-broken family scan (E8a): quantization 5.6e-16 over 51 gapped
   points (glide-only protection at scale); ζ₀ = ζπ persists ⇒ identity is
   not a K-consequence.
7. S_π optimization (E7c): 0.034 → 0.140 (wall cell) → 0.624 (±2 cells).
8. Part I compressed; abstract 218 words; Rydberg claim tempered with
   tolerance budget; bibliography verified against the published record
   (chen2020 = Zhao–Huang–Yang confirmed; dai2012 = JPA 45, 285301;
   li2015 authors corrected to Li–Mc Gettrick–Zhang–Zhang, pages added;
   higuchi2025 = Higuchi–Segawa; unused matsubara1955 removed).
9. Manuscript validation: mechanical checker
   (`code/part1_validation/check_manuscript.py`) — balanced environments,
   22/22 citations resolved, all \ref labels defined, all figure files
   present, math delimiters balanced, single intentional TODO (author
   email) — PASS; content audit — every review item present, every
   numerical claim in the text cross-checked against measured outputs —
   PASS. (pdflatex unavailable on this machine; compile check deferred.)

## Revision round 2 (2026-11-12, adversarial audit of manuscript v2)

The v2 manuscript was audited by testing its theorems directly rather than
re-running its scripts (which all reproduce). Seven errors were found, each
confirmed two ways (analytically and numerically), and all were fixed in
manuscript v3. Nothing below was taken from v2 without a new check.

**Errors found in v2 and their fixes**

1. **"Two invariants (ζ₀, ζ_π) from the two branch cuts" — wrong [FIXED].**
   Moving the branch cut gives H′ − π = H − π·sgn(H): the two flattened
   Hamiltonians are exact negatives, and det(−D) = det D for 2×2 blocks, so
   the two "invariants" are one number. The former open problem
   "ζ₀ = ζ_π" is this identity (E8a: 1e-15). Replaced by genuinely
   gap-resolved invariants from the half-step F (below).
2. **"Wall-mode existence is dictated by Δζ" — wrong [FIXED].** A strictly
   local unitary U → U·exp(iλ|m⟩⟨m|) (|m⟩ = wall mode, ±10 cells) leaves both
   bulk domains untouched; for 0.08 ≲ λ/π ≲ 0.90 the wall hosts no in-gap
   state, and at λ = π (real again) the mode sits at ε = 0 (E7d). The wall
   modes are protected by K (class D), not by the glide.
3. **"BDI enriched by the glide" — wrong [FIXED].** No k-independent chiral
   operator exists (E1f); the only constant symmetry is K: class D.
4. **Theorem 2 (winding) excluded the wrong lines [FIXED].** With
   R(θ) = e^{−iθσ_y/2} the gap closes on θ₁ ± θ₂ ∈ 2πℤ, not on
   θ₁ ± θ₂ ∈ {0, π}; θ₁ + θ₂ = π is gapped and θ₁ + θ₂ = 2π (gapless) was not
   excluded (V8). Restated with the correct set and an explicit bound:
   equality for all L > 2/g, g = min_k |sin ε(k)| (V8: 40 random points).
5. **Proposition 1 genericity condition wrong [FIXED].** The exceptional set
   is θ₂ ∈ π + 2πℤ (walk k-independent, Σ = σ_y admissible) and
   (θ₁, θ₂) ∈ (2πℤ)², not θᵢ ∈ {0, 2π} (V7b).
6. **Theorem 1 claimed K^Möb_L excludes k = π for every L [FIXED].** It
   contains k = π exactly when L is odd (V9).
7. **Overclaims removed:** "the unique way the standard walker admits it"
   (never proven); "persistent" Szegedy distinction (it is the same π/L
   shift in the rung-antisymmetric sector).

**Bibliography errors found and fixed** (every DOI re-checked against
Crossref): `asbothEdge2015` pointed to PRB 91, 195442, an unrelated graphene
paper (correct: PRA 91, 022324); `cedzich2018` merged two different papers
(AHP 19, 325 and Quantum 2, 95 — now two entries); `dai2012` had the wrong
authors (correct: P. Li and Z. Zhang); Quelle et al. is 2015 with DOI;
Zhou–Zhang–Pan is PRB 112, 134302 (2025). Added: Higuchi–Segawa orientable
paper (AIHPD, DOI 10.4171/AIHPD/206), Gross–Tucker, Asbóth 2012,
Asbóth–Tarasinski–Delplace 2014, Jiang et al. 2011, Cardano et al. 2017.

**New results (all machine-verified)**

| Claim | Check | Outcome |
| --- | --- | --- |
| Ũ = τ_z F̃†τ_z F̃; σ(b) = sin(ε/2), σ(d) = cos(ε/2); det a = conj det d; Möbius constraints; det Ũ₊₋ = 4 det b det d | E1j, 40 random walks (real + SU(2)) | PASS, ≤ 2e-15 |
| Gap-resolved (ζ₀, ζ_π): quantization with K and T broken | E9c, 264 gapped points | PASS, 3e-16; all four classes |
| ζ₀ jumps only at 0-gap closings, ζ_π only at π-gap closings | E9c (60 random paths), E3 (61×61), E8b (41×41) | PASS: own gap ≤ 0.04 on every jump step, other gap open |
| Closed form (real coins): det b(0) = (Y−X)/2, det d(0) = −(Y+X)/2 | E9a 543/543; E9f 5770/5770; E3 3286/3286; E8b 1470/1470 | PASS |
| ζ(U) = ζ₀ + ζ_π; H_eff parity = ζ₀ + ζ_π + 1 for both cuts | E9b 206/206; E3 3286/3286 | PASS |
| Conventions: F → iF flips both; frame change flips ζ₀ only | E9d 110/110 | PASS |
| T = ΓK, T² = e^{−ik}; Kramers pairs at k = π; cos ε(π) = −cos ᾱ | E1k, E9e | PASS, 4e-16; lifted for generic SU(2) coins |
| Wall rule N₀ ≡ Δζ₀, N_π ≡ Δζ_π (mod 2) | E4: 9 pairs × 3 widths | PASS 27/27 |
| Class-D Pfaffian parity of H_eff = ζ₀ + ζ_π + 1 | E9g 313/313 | PASS |
| Parity lemma (full shifts collapse) | E1i | PASS, commutant dim 4 |

**A correction caught during this round.** A first version of the E9b check
compared the half-zone *integers* n(U₊₋) and n₀ + n_π and failed at 63 of 206
points: the integers are defined only mod 2 (branch of θ(0)), so only the
parities may be compared. The check and the paper state parities.

**Not done / deferred.** The LaTeX has still never been compiled (no TeX on
this machine; fetching an engine was not permitted). The mechanical checker
and a command-level lint pass; compile on Overleaf before submission.

## Revision round 3 (2026-11-13): proof of the wall rule and robustness study

Addresses the five scientific issues and the numerical-robustness issues
listed after round 2. Manuscript v3 is preserved in
`paper/_backup_v3_2026-11-12/`.

### New theorems (manuscript v4)

1. **Closed form at all momenta (Proposition 2, extended).** With
   C = cos ᾱ, w = sin(k/2), A = C cos(β_A/2) cos(β_B/2):
   f₀(k) = A w² + (i/2)(1+C) sin β̄ · w + (Y−X)/2 and
   f_π(k) = A w² + (i/2)(1−C) sin β̄ · w − (Y+X)/2. Hence the **complete**
   gap-closing set: gap 0 closes iff X = Y (k = 0) or C = −1 (k = π, double);
   gap π iff X = −Y (k = 0) or C = +1 (k = π, double). Derived symbolically;
   checked at all k for 200 random points to 1.5e-15 and by 4000 random
   points for the absence of other closings (E10a).
2. **Wall rule (Theorem 5), now proven.** N_μ ≡ ζ_μ(p_L) + ζ_μ(p_R) (mod 2)
   for every real-coin wall. Proof: (i) parity lemma for real orthogonal
   operators plus interpolation of wall profiles; (ii) the μ-gapped region
   of the parameter torus has exactly two components (from 1.);
   (iii) one exactly solvable wall, between (π,0,0,0) and (π,0,π,π), where U
   is a signed permutation carrying one 4-cycle of sign +1 per wall, so
   N₀ = N_π = 1. The base case is reproduced in exact integer arithmetic for
   four ring sizes (E10b).
3. **Local real surgery (Proposition 3).** A local real orthogonal W keeps
   both parities if det W = +1 and flips both if det W = −1. Replaces the
   single-example claim of round 2. Checked on 200 random W (87 with det +1,
   113 with det −1, all as predicted, far wall never affected; E10f).

### Robustness results

| Issue raised | Study | Outcome |
| --- | --- | --- |
| Counts depend on thresholds | E11a: 108 (pair, tolerance, radius) combinations for pinned modes; 162 (window, radius) combinations for the in-gap census | Pinned count per wall **equals** (Δζ₀, Δζ_π), not merely mod 2, in all 108; census parities stable in all 162 (raw census counts do vary, e.g. C\|D 3–9) |
| Few system sizes | E11b: rings of 40–200 cells, sharp walls | One pinned mode per wall at every size; pinning error decays at the analytic rate (0.387, 0.567 vs 0.572, 0.708, 0.572 per cell) |
| Thin disorder statistics | E7a: 40 realisations per point, W up to 1.0 | Real disorder: deviation ≤ 5.8e-4 over 640 realisations, median ≤ 1.8e-5. K-breaking: median ∝ W^1.06 |
| Boundary audits resolution-limited | E11c: bisection on 40 random lines | Numerical jump sits on the analytic boundary to 2.7e-13 |
| ξ fit used 6 points | E7b/E10g: sharp walls, 14 separations, eight decades; E11d for the 0 mode | Fitted 0.7078 per cell vs analytic 0.7078 (ξ = 1.413); 0 mode 0.5724 vs 0.5724 |
| Profile dependence | E10c: 150 walls with random widths and random detours | 0 parity violations |
| Period shifts | E10d: 32 walls between a point and its 2π-shifted copies | all even |

### Richer family (E12)

Four-layer real half-step (eight angles): det F = 1, quantization 4e-16 on
323 gapped points, all four classes, parity = sign f(0), Kramers pairs,
gap-selective jumps, and pinned modes per wall = (Δζ₀, Δζ_π) on 91/91 sharp
walls. No class beyond the four.

**Negative/exploratory result, reported as such.** A search for a
symmetry-preserving gapped path joining the two pieces into which the
Kramers-doubled closings cut each class (F′ = F·W, W ∈ SO(4) in the unit
cell; ten parameters, twelve restarts per class) found none (best gaps
< 0.01). This suggests a finer invariant when T is present; it is listed as
an open problem, not claimed.

### Corrections caught during this round

- First versions of two checks failed for reasons of test design, not of
  the theory: the localisation fit included separations where the splitting
  had reached eigenvalue round-off (1e-13) and where hybridisation through
  the outer domain competes; and the elementary-crossing test used a pinning
  tolerance tighter than the inter-wall splitting close to a transition.
  Both were redesigned (fit restricted to splittings > 1e-11 and s ≤ 32;
  projector census on a 160-cell ring).
- A sentence in the proof of Proposition 2 claimed the f_π case follows from
  f₀ by "C → −C, Y → −Y"; the correct relation is f_π(k) = −f₀(−k) with
  C → −C (verified to 5e-16).
- The wording "gentle / non-gentle" attributed to Cedzich et al. could not be
  confirmed from their abstract; the manuscript now quotes their abstract
  ("sensitive to those compact perturbations which cannot be contracted to a
  trivial one") and states its own Proposition 3 independently.

## Revision round 4 (2026-11-14): adversarial audit of manuscript v4

Every proof was re-derived looking for gaps, the walk was re-implemented from
scratch, and every number quoted in the manuscript was compared with the
final logs. Five defects were found and fixed; none affects a theorem's
conclusion, but two were false statements.

1. **False remark after Theorem 4 [FIXED].** The text said that requiring F
   to be real removes the freedom to flip both invariants. It does not: the
   real replacement F → (𝕀⊗σ_x)F leaves U and det F unchanged and flips both
   (77/77 points, E9d). The pair (ζ₀, ζ_π) is attached to the half-step
   protocol; only ζ₀ + ζ_π and differences within a protocol family are
   intrinsic. The manuscript now says so.
2. **Proposition 1 fails for L = 1 [FIXED].** The cover has only 2L momenta;
   for L = 1 only e^{ik} = ±1 occur and Σ = σ_y commutes at any angles
   (commutant dimension 2 on the actual 2-cycle; 1 for L = 2, 3, 5, 8; V7c).
   The proposition now assumes L ≥ 2 and the proof uses the three distinct
   values of e^{ik}.
3. **Sign of a_z in Part I [FIXED].** The text gave a_z = −cos(θ₂/2) sin k,
   which belongs to the opposite Bloch convention from the one used in the
   proof of Proposition 1. In the paper's convention a_z = +cos(θ₂/2) sin k
   (checked to 2e-16). The winding theorem is unaffected (it uses |v′| only).
4. **A sentence about class A [FIXED].** "A class that is trivial in one
   dimension" is not safe for unitary walks, which carry a translation index.
   Replaced by the statement that no Altland–Zirnbauer invariant is available.
5. **Stale numbers [FIXED].** The surgery test counts were 88 and 112, not 87
   and 113 (the random stream shifted when an earlier loop was lengthened).

Also tightened: "generic real angles" for the no-collapse statements now
rests on 40 random points (E1f/g), not one; the photonic-platform sentence
now cites Schreiber et al., PRL 106, 180403 (checked against Crossref).

**Independent checks that passed.** A from-scratch real-space implementation
agrees with the library to machine zero on random inhomogeneous rings. A
flood-fill of the gapped regions on a 48³ grid of the parameter torus
confirms the two-component claim used in the proof of Theorem 5 for both
gaps. det d is invariant under random basis changes in its sector (6e-16).
Continuous real local rotations at a wall leave both modes pinned. Every
other number in the manuscript matches the final log.

## Open items carried into the paper as such (not overclaimed)

- A boundary signature of the glide invariants without K: impossible in 1D
  (no glide-preserving boundary); needs a 2D extension, not constructed.
- Index-theoretic classification of nonsymmorphic-chiral walks, and the
  finer invariant suggested by E12b.
- Theorem 5 beyond the two-layer family: step (i) holds verbatim and the
  numerics are complete (91/91), but the connectivity of the gapped regions
  is not known in closed form.
- The essential-spectrum statement and the localisation of the determinant
  argument in Proposition 3 are given at the usual level of rigour of the
  physics literature (finite ring with distant walls), not as an
  infinite-volume index theorem.
