# Space-glide quantum walk

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24336241.svg)](https://doi.org/10.5281/zenodo.24336241)

Code and data for the paper

> S. M. Yousuf Iqbal Tomal and Abdullah Al Shafin,
> *Möbius twists in discrete-time quantum walks: inert boundary conditions
> and gap-resolved glide invariants*, submitted to J. Phys. A: Math. Theor.
> (2026).

Every figure and every numerical statement in the paper is produced by the
scripts in this repository. Each statement in the paper cites a labelled
check (for example V5 or E9c); the tables below map the labels to scripts.

## Results in brief

**Part I — a Möbius twist in the boundary condition.** For the chiral
split-step walk the only compatible deck operators are the scalars ±𝕀
(outside an explicit exceptional set of coin angles). The anti-periodic walk
has the cylinder dispersion at half-odd-integer momenta, and its winding
numbers equal the cylinder's for every ring size L > 2/g, where
g = min_k |sin ε(k)|.

**Part II — a Möbius twist in the symmetry algebra.** The glide-chiral
operator Γ (a one-site translation times a coin flip) obeys
Γ(k)² = e^{−ik}𝕀, so its eigenbundle over the Brillouin zone is a Möbius
bundle. For every local half-step F, the walk U = ΓF†Γ⁻¹F is glide-chiral.
In the twisted eigenbasis, the blocks b and d of F have singular values
sin(ε/2) and cos(ε/2). Their determinants define two gap-resolved ℤ₂
invariants (ζ₀, ζ_π), protected by the unitary glide alone. For real coins:

- the phase diagram (four classes) and the complete set of gap closings are
  known in closed form;
- a glide time reversal T = ΓK with T² = e^{−ik} pairs the bands at k = π;
- domain walls bind N₀ ≡ Δζ₀ and N_π ≡ Δζ_π (mod 2) modes at quasi-energies
  0 and π. These modes are protected by the particle-hole symmetry K of real
  coins, not by the glide.

## Layout

```text
src/        spaceglide_walker.py  walk operators, invariants, closed forms
            figstyle.py           shared figure style
scripts/    one script per group of checks (table below)
figures/    generated figures (fig1-fig4 appear in the paper)
data/       raw data behind the phase-diagram figures
```

## Reproduce

```bash
pip install -r requirements.txt
python scripts/part1_canonical_mobius.py
python scripts/construction_checks.py
# ... any script in scripts/, from any working directory
```

Each script prints `[PASS]`/`[FAIL]` lines and, where relevant, writes its
figure to `figures/` and its data to `data/`. All scripts run on a laptop
CPU; the slowest (`four_layer_family.py`) takes about ten minutes.

| Script | Labels | Contents | Output |
| --- | --- | --- | --- |
| `part1_canonical_mobius.py` | V1–V9 | deck operators, half-odd-integer sampling, winding equality and its bound, Möbius ladder | — |
| `construction_checks.py` | E1a–E1k | glide-chiral construction, no-collapse tests, parity lemma, half-step block identities, antiunitary symmetries | — |
| `bands_and_winding.py` | E2 | bands and half-step winding data | Fig. 1 |
| `gap_resolved_invariants.py` | E9a–E9g | closed form, relation to the Floquet operator, glide-only protection, conventions, Kramers pairs, class-D Pfaffian | Fig. 2 |
| `phase_diagram_slice.py` | E3 | gap-selective audit on an (α_A, α_B) slice | Fig. S1 |
| `domain_walls.py` | E4–E6 | wall census for all class pairs, scalar twist on the glide walk, return amplitudes | Fig. 3 |
| `wall_mode_protection.py` | E7a–E7d | disorder statistics, hybridisation, signal optimisation, local surgery | Fig. 4 |
| `branch_cut_and_second_slice.py` | E8a–E8b | branch-cut identity, second slice | — |
| `wall_theorem_checks.py` | E10a–E10g | each step of the proof of the wall theorem | — |
| `robustness.py` | E11a–E11d | thresholds, system size, boundary location, localisation lengths | — |
| `four_layer_family.py` | E12a–E12b | the theory in a four-layer family | — |

## Licence and citation

MIT licence (see `LICENSE`). Version 1.0.0, the code behind the paper's
results, is archived on Zenodo at
[doi:10.5281/zenodo.24336241](https://doi.org/10.5281/zenodo.24336241); the
DOI [10.5281/zenodo.24336240](https://doi.org/10.5281/zenodo.24336240) always
resolves to the latest version. Citation metadata are in `CITATION.cff`.
