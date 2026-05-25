# cite_finder test cases — v2

Builds on `test_cite_cases.md` with more papers covered (12 new positives,
6 new negatives), more variety in claim difficulty (verbatim numbers,
compound claims, multi-paper disambiguation, wrong-direction tests), and
edge cases that exercise the robustness layers (reranker + diversification
cap + sentence-level entailment + citation-as-evidence rule).

Each ground-truth sentence is drawn from a chunk I actually saw in your
library during this session's debug runs — they're real, not guesses.
That said, the EXPECTED paper/page is still worth a sanity check before
declaring a test "failed" — your library may have more than one paper
containing similar facts.

How to read this file:

- **Easy** = paraphrase shares most distinctive vocabulary with source.
- **Medium** = paraphrase reworked but core fact unchanged.
- **Hard** = significant semantic transformation, or compound claim
  spanning multiple sentences.
- **Negative** = expected to return empty / very low confidence.
- **Multi** = multiple papers in your library could legitimately support
  the claim; checks which one the system picks first.

| # | Claim to paste | Expected paper | Expected page | Ground-truth sentence | Difficulty |
|---|---|---|---|---|---|

## New positive cases

| 11 | A photonic lantern's fundamental multimode-end mode maps almost perfectly to a single SMF core at the output when the device is optimised for symmetry. | paper 14 (Leon-Saval 2018) | 11 | "The fundamental mode of the multimode end of the lantern has a near perfect correspondence to a single SMF core at the output based on symmetry and an optimised device design." | Easy |
| 12 | An extreme AO system providing 90% Strehl can achieve up to 67% fibre coupling efficiency in the 1500–1600 nm range. | paper 14 (Leon-Saval 2018) | 9 | "With ExAO levels of wavefront correction (90% Strehl ratio) a coupling eﬃciency as high as 67 ± 2% could be achieved in the range of 1500–1600 nm." | Easy |
| 13 | Polarization noise in single-mode fibre spectrographs ranges from tens of cm/s to several m/s and is exacerbated by polarization-sensitive spectrograph optics. | paper 16 (Bechter 2020) | 1 | "We ﬁnd that polarization noise is a tens of cm s−1 to several m s−1 effect, which is exacerbated by the degree of polarization of the light source and the polarization response of the spectrograph optics." | Easy |
| 14 | Seeing-limited high-resolution spectrographs grow in volume following a D³ power law with telescope diameter, due to conservation of beam etendue. | paper 25 (Mawet 2022) | 2 | "Seeing-limited high-resolution spectrographs, by virtue of the conservation of beam etendue, grow in volume following a D3 power law (D is the telescope diameter)..." | Easy |
| 15 | Adding PIAA optics to the KPIC Phase II system reduces the L-band detection exposure time by about 33%, from roughly 3 to 2 hours. | paper 22 (Calvin 2021) | 11 | "the PIAA optics offer the advantages in exposure time, reducing the exposure time of the expected phase II system by ∼33% from 3 to 2 hr" | Medium |
| 16 | Observing in the infrared reduces "astrophysical jitter" from rotating star spots by a factor of several compared to visible-light instruments. | paper 36 (Crepp 2014) | 2 | "by working in the infrared, 'astrophysical jitter' caused by star spots that rotate in and out of view as a star spins is reduced by a factor of several compared to visible light instruments." | Medium |
| 17 | Telescope-environment effects — bending, twisting, and thermal stress — induce time-varying birefringence in single-mode fibres. | paper 16 (Bechter 2020) | 1 | "Conditions at a telescope observatory will subject the ﬁber to mechanical (bending and twisting) and thermal stresses, inducing birefringence that varies in time." | Medium |
| 18 | Although single-mode fibres deliver a perfectly scrambled output beam, they still propagate two orthogonal polarization modes. | paper 10 (Halverson 2015) | 1 | "While these fibers can produce a stable and perfectly scrambled output illumination, typical SMFs do support two fundamental polarization modes." | Medium |
| 18-note | (Originally expected Restori 2024 p.3 — that was a contaminated test entry. The sentence above actually lives in Halverson 2015 p.1; we found this when cite_finder refused to confirm the misattribution from an earlier RAG-hallucinated response. cite_finder catching the test author's own bug, basically.) | — | — | — | — |
| 19 | The peak fibre-coupling efficiency reached in lab measurements approaches 74% when the wavefront error is held very low, in the 1500–1600 nm range. | paper 13 (Jovanovic 2016) | 5 | "the coupling eﬃciency is maximum at longer wavelengths (1500-1600 nm) and reaches a value as high as 74% in the limit where the wavefront error is low" | Medium |
| 20 | The 50% gain in IR-WFS sky coverage becomes much more dramatic in obscured regions such as star-forming clouds, where the population is dominated by red M and late-K stars. | paper 31 (Mawet 2016) | 3 | "In obscured areas such as SFR, the gain is much more dramatic. The population of young stars in Taurus, 140 pc away, is dominated by M stars and very late K stars,13 making IR WFS essential for these very red stars." | Hard (compound) |
| 21 | KPIC Phase II added a 1000-element deformable mirror to better correct static aberrations and control speckles at the fibre location. | paper 29 (Jovanovic 2025) | 2 | "The first upgrade was a 1000-element deformable mirror (DM), which allows for superior static aberration correction and speckle control at the location of the fiber." | Medium |
| 22 | KPIC Phase I has already detected more than 23 exoplanets and brown dwarfs since its 2018 commissioning. | paper 29 (Jovanovic 2025) | 2 | "That mini- malist version of KPIC was commissioned between 2018 and 2020 and proved to be highly capable, successfully detecting 23+ exoplanets and brown dwarfs to date." | Hard (specific number + date) |

## Multi-paper / disambiguation

These should return a hit — but multiple papers in the library can support the claim.
Useful for checking which paper the system ranks first and whether the diversification
cap surfaces multiple options.

| M1 | PIAA coronagraphs combine achromaticity with a small inner working angle. | papers 4, 22, 26, 28 (Guyon / Calvin / Martinez / Restori — all touch PIAA) | varies | varies | Multi |
| M2 | Diffraction-limited spectroscopy is essential for ELT-scale instruments because it decouples spectrograph size from telescope aperture. | papers 13, 25 (Jovanovic / Mawet — both make this argument) | varies | varies | Multi |
| M3 | Single-mode fibres eliminate modal noise that plagues multi-mode fibre feeds. | papers 10, 16, 36 (Halverson / Bechter / Crepp — all establish this) | varies | varies | Multi |

## Negative controls

These should return either no hit, or only low-confidence hits (≥0.7 = false positive).

| N3 | HARPS-N achieves radial velocity precision better than 0.5 m/s on cool dwarfs. | empty (no HARPS-N paper in library) | — | — | Negative |
| N4 | Sky coverage with an IR wavefront sensor is typically 80% higher than with a classical visible WFS. | empty or low-conf (real number is 50%, not 80% — should reject the wrong-value version) | — | — | Negative (wrong number) |
| N5 | The James Webb Space Telescope's NIRSpec achieves R=2700 in mid-resolution mode. | empty (no JWST paper in your library) | — | — | Negative |
| N6 | Bessel beam propagation distance scales linearly with input power. | empty (the actual claim in Chu 2015 is that distance is SHORT due to LOW power — opposite direction) | — | — | Negative (wrong direction) |
| N7 | Optical fibres can transmit data at speeds exceeding 100 Gbps. | empty (telecom claim, not in your astronomy library) | — | — | Negative |
| N8 | Single-mode fibres have core diameters of 50 μm. | empty (the real value is ~9 μm; this is a multi-mode core diameter) | — | — | Negative (wrong number) |

## What each test is probing

- **11, 13, 14, 18, 21, 22** — single-paper accuracy on real distinctive sentences. Should all hit at ≥0.85.
- **12, 19** — both point at Leon-Saval / Jovanovic 2016 with overlapping 1500-1600 nm coupling stats. Tests whether the system picks the more specific match.
- **15, 16, 17** — verify that the previously-tricky papers (Calvin 2021, Crepp 2014, Bechter 2020 birefringence) still resolve correctly with the latest changes.
- **20** — compound claim with two facts. Tests sentence-level entailment on a passage spanning two sentences.
- **M1, M2, M3** — verifies the diversification cap shows multiple candidate papers when several legitimately support a claim.
- **N3, N5, N7** — claims about subjects truly absent from your library; should be empty.
- **N4, N6, N8** — claims that share vocabulary with library content but assert WRONG facts (wrong number, wrong direction). The citation-as-evidence rule + entailment confidence threshold should reject these.

## Recommended run order

1. **Tests 11-22 first** — confirms positive accuracy with the latest robustness layers.
2. **M1-M3** — observe multi-paper behavior with the diversification cap.
3. **N3-N8** — confirm no false positives on negatives.

If anything from tests 11-22 fails, re-run with `--debug` and we can
look at the entailment table for that case.
