# REM-θ-framework — Lorentz Boost θ-Parametrization & Gravitational Interferometry

Companion simulation code for the article *"Relativistic electromagnetic mass:
applications to the Lorentz boost and gravitational quantum interferometry"*
(Article 2 of the REM-θ-framework).

This repository contains two independent, self-contained simulation sets:

| Folder | What it validates |
|---|---|
| [`lorentz_boost/`](lorentz_boost/) | The θ-parametrization of the Lorentz boost tensor (γ = sec θ), validated against the standard formulation for five alternative physical inputs (velocity, total energy, kinetic energy, momentum, de Broglie wavelength) |
| [`gravitational_interferometry/`](gravitational_interferometry/) | The geometric reformulation of gravitationally-induced quantum phase shift (Δφ = 2πgA·cotθ·sinδ / c²λ_C), verified against the standard formula for neutron interferometry (COW effect, real experimental data), electron interferometry, and the photon gravitational redshift (Pound–Rebka) |

## Important: proof of concept, not experimental validation

The algebraic identity between the θ-parametrization and the standard formulation
is exact (verified to machine precision, error < 10⁻¹⁵). This means the
simulations will necessarily agree with standard physics wherever standard
physics has already been experimentally validated — **this is expected by
construction, not an independent experimental test of the underlying
REM-θ-framework**. These scripts are proof-of-concept and internal-consistency
checks, not experimental validation of the framework itself.

## Requirements

```
Python >= 3.9
numpy >= 1.21
matplotlib >= 3.5
scipy
```

Install with:

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# Lorentz boost validation, benchmark, and figures
cd lorentz_boost
python lorentz_theta_complete.py

# Gravitational interferometry (COW / electron / photon)
cd ../gravitational_interferometry
python cow_simulation_final.py
```

See each folder's section below for details and command-line options.

---

## `lorentz_boost/`

Validates five alternative physical inputs for the θ-parametrization of the
Lorentz boost (θ = arcsin β, arccos(E/E₀), arcsec(1+K/E₀), arctan(pc/E₀),
arctan(λ_C/λ_dB)):

- **`lorentz_theta_complete.py`** — full validation suite, computational
  performance benchmark, and figure generation. Produces 4 figures:
  1. Algebraic equivalence and numerical stability across 9 decades of γ
  2. Computational performance (speedup and error), median of 15 paired
     trials with warm-up excluded
  3. 4×4 Lorentz tensor heatmaps (standard vs. θ-parametrization)
  4. Validation errors for 6 boost directions × 3 velocities — explicitly
     testing the θ-parametrization tensor Λ(θ), not the standard boost

  ```bash
  python lorentz_theta_complete.py [--n 1000000] [--outdir ./figures] [--quiet]
  ```

**Note on benchmark numbers:** speedup and timing results are
hardware-dependent. Re-run `lorentz_theta_complete.py` on your own machine
before citing specific numbers — do not assume the values in the article
match your hardware.

---

## `gravitational_interferometry/`

**`cow_simulation_final.py`** — generates the combined figure comparing the
θ-parametrization's geometric reformulation of gravitationally-induced
quantum phase shift against the standard quantum-mechanical formula, for
three systems:

- **Neutron interferometry (COW effect)** — validated against real
  experimental data digitized from Fig. 11 of Staudenmann et al. (1980),
  *Phys. Rev. A* 21, 1419. The background/contrast envelope fit follows the
  full 9-parameter phenomenological form of Eq. 54 in that paper.
- **Electron interferometry** — theoretical prediction only; no discrete
  experimental dataset is available for this configuration.
- **Photon gravitational redshift** — compared against the single
  experimental point from Pound & Rebka (1959).

```bash
python cow_simulation_final.py
```

Output: `figures/cow_fig_neutron.png`, `figures/cow_fig_electron_photon.png`,
and `cow_results.txt` (numerical summary).

### Data

`data/Staudenmann_Fig11.csv` contains 193 points digitized via
WebPlotDigitizer from Fig. 11 of Staudenmann et al. (1980) — see the file
header for full provenance and citation. If this file is missing or moved,
the script prints a warning and falls back to synthetic demonstration data
rather than failing silently; check the console output and the panel
legend/title to confirm which one was used.

---

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
├── lorentz_boost/
│   ├── lorentz_theta_complete.py
│   └── figures/            (generated at runtime)
└── gravitational_interferometry/
    ├── cow_simulation_final.py
    ├── data/
    │   └── Staudenmann_Fig11.csv
    └── figures/             (generated at runtime)
```

## Attribution

- **Theoretical framework, physical concepts and hypotheses:** Luís Gustavo
  Costa Reis
- **Computational implementation, validation, and figure generation:**
  Claude (Anthropic), under the author's direction

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the article once published;
a Zenodo DOI for this code will be added here once minted.

## License

Code in this repository is released under the MIT License (see
[`LICENSE`](LICENSE)) unless noted otherwise. The digitized dataset in
`gravitational_interferometry/data/` derives from a copyrighted published
figure (American Physical Society) and is included for academic
reproducibility only — see that file's header for details.
