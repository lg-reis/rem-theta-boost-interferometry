# Changelog

## 1.8.0 — 2026-10-22

### Documentation and terminology
- "Tensor" replaced by "transformation matrix" in comments, docstrings,
  console output and `lorentz_results.txt` labels; the command-line option
  is now `--nmatrix` (`--ntensor` is still accepted).
- README and module docstring: six inputs (time dilation added to the list),
  figures listed in the order they appear in the article (A2, A3, A1, A4).
  No number, fit or figure changed.

### `lorentz_boost/lorentz_theta_complete.py`
- Benchmark rewritten so that all inputs are on equal footing: each input
  starts from the raw physical quantity (v, E, K, pc, λ_dB, t) and performs
  its own normalization inside the timed region (before, pc/E₀ and
  λ_C/λ_dB were pre-normalized while E and K were divided inside the timer,
  and β was given ready to the standard formula).
- Every mode now returns γ = sec θ and βγ = tan θ, and the error of both is
  reported (before, only γ was checked).
- Each input is timed along two paths: the arithmetic form derived from the
  θ identities and the explicit θ path (arcsin/arccos/arctan followed by
  1/cos θ and tan θ).
- Errors measured in two ranges: β ∈ [0.01, 0.9999] (2000 samples) and
  β ∈ [10⁻⁸, 10⁻²] (500 samples).
- Trials run in random order in each repetition.
- Fig. A2 redrawn with three panels: (a) speedup, arithmetic vs θ path;
  (b) and (c) max relative error of γ and βγ in the two ranges.
- Section [3] of `lorentz_results.txt` rewritten accordingly.

- Fig. A3 and Fig. A4 titles: "tensor" replaced by "transformation matrix";
  A4 title no longer carries "(this work) — not the standard boost".
- Fig. A3: exactly-zero differences are labeled "0" (the 1e-18 offset is
  used only for the log color scale).
- Fig. A4: a single shared legend below the panels (it no longer covers
  bars).

### `gravitational_interferometry/cow_simulation_final.py` (v9)
- Side-panel curve labels clarified so that no curve reads as "the
  framework disagreeing with the data": green = "q_grav only (no bending
  corr., ideal contrast)"; blue = "θ-form + bending corr."; red = "Standard
  form + bending corr. (identical to θ-form)".
- λ_dB written with an upright subscript (\rm dB) in the equation line and
  subtitle.
- References to "Eq. 4.4" replaced by "geometric θ-form (Sec. 4)", since the
  equation number changes in the revised article.
- Side-panel spacing adjusted for the two-line labels. No number, fit or
  data changed.

## 1.7.0 — 2026-10-14

### `lorentz_boost/lorentz_theta_complete.py`
- Fig. A1(a) legend: the two dashed-curve entries now read
  `$\sec\theta$ ($\theta=\arcsin\beta$)` and
  `$\tan\theta$ ($\theta=\arcsin\beta$)` (was `($\arcsin\beta$)`, no
  "$\theta=$"), completing the "$\theta=$" standardization started in
  1.6.0 for panel (b) and Fig. A2 — now every labeled entry across both
  figures explicitly names the angle $\theta$ rather than just the arc
  function. No data, color, or linestyle changed — label text only.

## 1.6.0 — 2026-10-14

### `lorentz_boost/lorentz_theta_complete.py`
- Fig. A1(b) and Fig. A2 (both panels): every entry's label now reads
  "$\theta=\dots$" (e.g. `1: $\theta=\arcsin(\beta)$`), not just input #6.
  Before, only input #6 (time dilation) carried the explicit "$\theta=$",
  which made it look like the odd one out and risked reading the arc
  functions themselves as the Lorentz factor instead of the angle $\theta$
  that feeds $\gamma=\sec\theta$. The "Standard" entry is unchanged (it
  computes $\gamma$ directly, with no angle involved).
- Fig. A2's x-axis tick labels are now rotated 30° (was 15°, font size
  8pt) since the added "$\theta=$" made every label longer; this removed
  the overlap between adjacent labels ("Standard" running into entry 1)
  that the longer text caused at the old rotation.
- No input, number, color, or linestyle was touched — only label text and
  tick rotation/font size.

## 1.5.0 — 2026-10-13

### `lorentz_boost/lorentz_theta_complete.py`
- Fig. A1(b) (equivalence/stability): inputs #4 and #5 were still labeled
  with the arithmetic/square-root form ($\sqrt{1+(pc/E_0)^2}$,
  $\sqrt{1+(\lambda_C/\lambda_{\rm dB})^2}$) left over from before the
  1.4.0 relabeling pass. Changed to the trigonometric form
  ($\arctan(pc/E_0)$, $\arctan(\lambda_C/\lambda_{\rm dB})$), matching
  Fig. A2's labels and the article's Table 1 convention (already correct
  there). Only the label text changed — the underlying computation
  (`s4`, `s5`), colors and linestyles are untouched.
- No other input, number, or figure panel was touched.

## 1.4.0 — 2026-10-12

### `lorentz_boost/lorentz_theta_complete.py`
- New input #6 for the θ-parametrization: **time dilation** ($\sec\theta =
  t/\tau$, $\theta=\mathrm{arcsec}(t/\tau)$). Added to `exact_inputs()`,
  `exact_inputs_from_beta()`, `sec_methods()`, Fig. A1 (equivalence/
  stability) and Fig. A2 (benchmark); the `[3] Benchmark` section of
  `lorentz_results.txt` picks it up automatically. No other input, figure
  panel or number was touched.
- This input is numerically identical to input #2 (total energy $E/E_0$):
  both equal $\gamma$ directly by definition ($\sec\theta=\gamma=E/E_0=
  t/\tau$), so their error and stability columns match exactly — documented
  in the code and in the article's Table 1 footnote, not a bug.
- Figure labels for inputs #2 and #3 now show the trigonometric form
  ($\arccos(E_0/E)$, $\mathrm{arcsec}(1+K/E_0)$) instead of the bare ratio,
  matching the article's Table 1 convention and the new input #6's label.
- Fig. A1(b) legend: more vertical spacing between entries (was cramped,
  "Standard" and "1: arcsin(β)" nearly touching).
- `lorentz_results.txt` benchmark section: LaTeX artifacts in the plain-text
  labels cleaned up (`\mathrm{arcsec}(...)` and `{\rm dB}` were leaking
  braces into the printed table; harmless but untidy).

### Article text (`Boost_Lorentz_-_ArXiv_v2.tex`)
- Table 1 (`tab:entradas_theta`): added row 6 (time dilation), with the
  speedup measured in this round (~0,86×) and a footnote sentence noting
  it shares input #2's numbers by definition. Rows 1–5 and their numbers
  are untouched (still from the earlier reference run — re-run and
  substitute as usual, see the `[RERUN]` comment above the table).
- No other count of "how many inputs" in the prose (abstract, Fig. A1/A2
  captions, conclusions) was changed — by request, those are being updated
  manually.

## 1.3.0 — 2026-10-07

### `gravitational_interferometry/cow_simulation_final.py`
- All plot labels translated to English ("Curves" / "Fit results" side-panel
  titles, "Experimental data", "Standard QM + corr. (identical)", $q_{\rm
  free}$; these were still in Portuguese after the 1.2.0 side-panel
  redesign). No other change — same numbers, same `cow_results.txt`.

### `lorentz_boost/lorentz_theta_complete.py`
- Checked end to end for leftover Portuguese text; none found (comments,
  labels and printed output were already in English).

## 1.2.0 — 2026-10-07

### `gravitational_interferometry/cow_simulation_final.py`
- New figure layout: the main plot (oscillation + residuals) is now narrower
  and shorter, and a single white side panel ("Curves" + "Fit results",
  both bold titles, left-aligned, same width) replaces the in-plot
  legend and yellow info box that were overlaid on the data before. The
  figure goes from tall-narrow to wide-short, which fits article pagination
  better.
- Fixed a header-spacing bug in the first version of this layout (the
  equation line could overlap the subtitle beneath it); figure height and
  text y-positions adjusted.
- No changes to the physics, the fit, or the numbers reported — same
  `cow_results.txt` content as 1.1.0.

### Article text (`Boost_Lorentz_-_ArXiv_v2.tex`)
- Fig. 5 caption shortened further: since the legend and fit numbers now
  live inside the figure itself, the caption only states what the plot
  shows, the data source, and points to the script / `cow_results.txt` for
  reproduction details.

## 1.1.0 — 2026-09-19

### `lorentz_boost/lorentz_theta_complete.py`
- Benchmark baseline: the standard formula now computes the square root once
  (the earlier baseline evaluated it twice, which inflated all speedups). The
  speedup with respect to the earlier baseline is still reported in
  `lorentz_results.txt`.
- Errors are measured against a 60-digit reference (`decimal`), with inputs
  computed exactly and rounded to float64. Fig. A1(b) and Fig. A2(b) now include
  the standard formula.
- New random-tensor test (10⁶ boosts, random directions) and the residual
  scaling with γ; the earlier "4.4×10⁻¹⁶" holds for β ≤ 0.7 only.
- Group test in Fig. A4 now composes collinear boosts along each direction (it
  used the x axis for every direction); the second velocity is defined by a
  single rule shared by figure and terminal table.
- The standard tensor in Fig. A3 is written independently of the θ tensor.
- θ → 0: (γ−1) evaluated as tan²θ/(1+sec θ); tan θ from E or K uses
  √(x(x+2)) forms (stable near γ = 1).
- New studies: float32 (β vs p input) and the composition rule
  tan(π/4 + θ/2).
- `boost_theta` requires a unit direction vector.
- Writes `lorentz_results.txt` (speedups are in its "[3] Benchmark" section);
  removed `plt.show()` and dead code.

### `gravitational_interferometry/cow_simulation_final.py`
- Adds a free-q fit, residual diagnostics (reduced χ², Durbin–Watson, lag-1
  autocorrelation), the fit-parameter table and the RMS/amplitude definition
  (half of max − min of the data).
- Multi-start fit (seeded), keeping the original starting point first.
- Scope reduced to the neutron (COW) case. Photons (Eq. 4.5, θ = π/4 and the
  Pound–Rebka panel) were removed because they are not an algebraic equivalence
  with the standard formula; the electron panel was removed because it was a
  pure identity test with no data. Both are left as future work.
- The single figure is `figures/cow_fig_neutron.png`; the info box now also
  shows θ ≈ 1.2×10⁻⁵ rad (essentially non-relativistic regime).
- Identity tolerance tightened to 10⁻¹⁴ (measured ~10⁻¹⁶); figure label
  updated accordingly.
- Paths relative to the script folder; results file written next to the script.
- Docstrings corrected (outputs, non-relativistic reference expression).

### Article text (`Boost_Lorentz_-_ArXiv_v2.tex`)
- Removed all remaining mentions of photons and electrons (future-work
  clauses in the conclusions); scope is now strictly the Lorentz boost and
  the neutron COW interferometry.
- Removed Table 2 (the 8 fit parameters with bounds): they are documented
  only in the script's `cow_results.txt`, not in the article body.
- Shortened the Fig. 5 (COW) caption: removed the interferometer geometry
  (a, d) and the explicit 9-parameter formula, kept λ₀ and the four
  diagnostic numbers the text relies on (RMS %, χ²_red, Durbin–Watson,
  q_free).

### Repository
- README, CITATION and `.gitignore` updated; this changelog added.
