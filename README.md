# Resonant Scaling Theory (RST) v6.1

**A topological framework for neural recovery after axonal shearing**

RST models brain recovery not only as biological repair, but as a **topological optimisation problem**.  
Two recurring invariants appear in the data:

- **6.1-second ignition delay** — a characteristic bottleneck when neural clusters must re-synchronise  
- **39 Hz resonance** — a high-frequency stability anchor linked to reduced topological friction

**Origin**: Direct clinical observation of a colleague (Subject 0284) recovering from axonal shearing, later examined against 581 clinical EEG files from the Temple University Hospital EEG (TUH-EEG) Corpus.

**Status**: Active independent research · Apache 2.0 · Open for collaboration and independent replication

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18898758.svg)](https://doi.org/10.5281/zenodo.18898758)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

---

### Quick Links
- [Preprint (Zenodo)](https://doi.org/10.5281/zenodo.18898758)
- [Core Parameters](#core-parameters)
- [Theoretical Overview](#theoretical-overview)
- [Limitations](#limitations)
- [How to Contribute / Contact](#collaboration)

---

## Core Parameters

| Parameter                  | Value                  | Meaning |
|---------------------------|------------------------|--------|
| Ignition Delay            | **6.1 s**              | Characteristic topological friction / state-transition bottleneck |
| Stability Constant        | **39 Hz**              | Target frequency associated with cluster re-synchronisation |
| Power-law correlation     | R² = 0.1789 (p < 0.001)| Observed across N=581 TUH-EEG files |
| Critical Coupling (RSN)   | ≈ 1.0 (±0.05)          | Optimal recovery window |
| Subject 0284 baseline RSN | 0.98                   | Within critical coupling range at study closure |

**Resonance Stability Number (RSN)**  
- RSN < 1 → under-coupled  
- RSN ≈ 1 → critical / optimal  
- RSN > 1 → over-coupled / potentially unstable

---

## Theoretical Overview

RST treats the injured brain as a high-dimensional graph:

- Vertices = neural clusters  
- Edges = axonal pathways  
- Key metric = Algebraic Connectivity (λ₂) of the graph Laplacian  

Axonal shearing reduces λ₂ (weaker network cohesion). The **6.1-second window** is interpreted as the time required for the damaged graph to find a viable re-synchronisation pathway. The **39 Hz** component is proposed as a stability frequency that helps minimise topological friction during this transition.

The same 6.1 s timescale has been observed in two different disruption types:
- Chronic structural damage (axonal shearing – Subject 0284)
- Acute hemodynamic disruption (tussive syncope case)

This cross-mechanism consistency is one of the central observations of the work.

An exploratory **5:1 phase-locking hypothesis** relating 39 Hz to the Schumann resonance fundamental (\~7.83 Hz) is also presented, with explicit falsifiability criteria. This remains a hypothesis requiring further experimental testing.

---

## Repository Contents

- Full theoretical documentation and mathematical appendices  
- Analysis and simulation scripts (Python)  
- Session logs and validation notes  
- Preprint PDF (also available on Zenodo)

> Note: Some toolkit scripts and full TUH analysis summaries referenced in earlier versions are still being organised for public release. The preprint currently contains the complete formal documentation.

---

## Limitations (Important)

- Primary longitudinal characterisation is based on N=1 (Subject 0284) plus supporting observations.  
- Broader cross-subject and healthy-control validation is still needed.  
- The 5:1 Schumann relationship is correlational; causal mechanisms have not been demonstrated.  
- All analysis to date was performed with accessible tools (including mobile Termux environments). Institutional replication is encouraged.  
- Schumann field amplitudes are extremely weak (picoTesla range); any coupling would be resonant rather than amplitude-driven.

These limitations are stated explicitly so the work can be tested and improved.

---

## Collaboration

RST is open-source and actively seeking:

- Researchers with EEG / TBI datasets  
- Clinicians working in neural recovery  
- Independent replication attempts  
- Critical feedback on the mathematical and empirical claims

**Contact**  
JJ Botha  
Independent Researcher – Nelspruit, Mpumalanga, South Africa  
Email: jjbotha1991@gmail.com  

---

## Citation

```bibtex
@misc{botha2026rst,
  author       = {Botha, JJ},
  title        = {Scale-Analogous Coupling in Neural Recovery: 
                  A Topological Framework for Post-Axonal Shearing Stabilisation},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18898758},
  note         = {Resonant Scaling Theory v6.1. Apache 2.0}
}