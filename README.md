Resonant Scaling Theory (RST) v6.1
Scale-Analogous Coupling in Neural Recovery: A Topological Framework for Post-Axonal Shearing Stabilisation
Author: JJ Botha | Independent Researcher | Nelspruit, Mpumalanga, South Africa
Contact: jjbotha1991@gmail.com
License: Apache 2.0
Preprint: Zenodo https://doi.org/10.5281/zenodo.18898758
Status: Active Research | Open for Collaboration
📌 Overview
Resonant Scaling Theory (RST) v6.1 models neural recovery following axonal shearing not as a purely biological repair process, but as a topological optimisation problem within a high-dimensional graph.
The framework proposes that the injured brain navigates recovery through a characteristic fast-slow coupling motif, utilising specific high-frequency stability anchors to overcome topological friction and re-synchronise disconnected neural clusters.
The theory originated from direct clinical observation of a colleague (Subject 0284) recovering from axonal shearing, and was subsequently validated against 581 clinical EEG files from the Temple University Hospital EEG (TUH-EEG) Corpus. All tools are designed for accessibility — runnable on a mobile phone via Termux with no institutional hardware required.
"The bridge holds not because it was never broken, but because it learned to resonate."
📊 Core Parameters & Empirical Findings
Batch analysis of 581 clinical EEG files identified a universal instability threshold governing state-transitions in injured neural networks.
Parameter
Value
Clinical Significance
Ignition Delay
6.1\text{ s}
High-dimensional topological friction index
Stability Constant
39\text{ Hz}
Target frequency for neural cluster re-synchronisation
Power-Law Correlation
R^2 = 0.1789\ (p < 0.001)
Instability threshold across N=581 subjects
Alpha Anchor
8.79\text{ Hz}
Subject 0284 individual alpha frequency (IAF); baseline coherence
Critical Coupling
\mathcal{R}_{sn} = 1\ (\pm 0.05)
Optimal recovery state indicator
Subject 0284 Baseline RSN
\mathcal{R}_{sn} = 0.98
Within critical coupling window at study closure
The Resonance Stability Number (RSN)
RSN Value
System State
Clinical Interpretation
RSN < 1
Under-coupled, damped
Under-stimulated, depressed, low connectivity
RSN = 1
Critical coupling
Optimal recovery, insight, cognitive flow
RSN > 1
Runaway, chaotic
Over-stimulated, seizure risk, manic state
All spectral power values computed via Welch's method (2-second sliding window, 50% overlap) on bandpass-filtered EEG signals. Replicable with MNE-Python, EEGLAB, or equivalent.
🧠 Theoretical Framework
The Neural Network as a Topological Graph
RST models the brain as a weighted graph G = (V, E) where:
V = neural clusters (vertices)
E = axonal pathways (edges)
L = D - A = Laplacian matrix (D = degree matrix, A = adjacency matrix)
Algebraic Connectivity \lambda_2 (second-smallest eigenvalue of L) is the primary stability metric. Axonal shearing reduces \lambda_2, indicating reduced network cohesion.
The Fiedler Vector (eigenvector associated with \lambda_2) identifies the structural bottleneck nodes — the Bridge Nodes — most critical to network integrity and most responsive to targeted stimulation.
Topological Friction and the 6.1-Second Window
Topological friction is the measurable resistance experienced by high-dimensional signals traversing a damaged network. This manifests as a consistent 6.1-second Ignition Delay — the time required for the neural graph to search for a stable state-transition pathway following disruption.
Crucially, this delay has been observed across two independent disruption mechanisms:
Feature
Subject 0284 (Axonal Shearing)
Case Study Alpha (Tussive Syncope)
Disruption type
Structural / chronic
Hemodynamic / acute
Mechanism
Axonal pathway degradation
Cerebral hypoperfusion (Valsalva)
Laplacian impact
Permanent edge-weight reduction
Transient full zeroing (\lambda_2 \to 0)
Ignition Delay
6.1s (500+ iterations)
~6.1s (longitudinal observation)
Recovery marker
39 Hz spectral peak lock
Snap-to lucidity
RSN trajectory
0.61 → 0.98 (months)
0 → 1 within 6.1s window
This cross-mechanism convergence supports the RST claim that the 6.1-second bottleneck is a fundamental property of human neural graph state-transitions, not a pathology-specific artefact.
🏥 Clinical Implementation
Component
Technical Definition
Clinical Application
6.1s Bottleneck
High-dimensional topological friction
Explains ignition delay in post-injury cognitive processing
Spectral Metric
Algebraic Connectivity \lambda_2
Measures structural integrity of remaining neural bridges
Fiedler Vector
Partitioning eigenvector of Laplacian
Locates Bridge Nodes requiring targeted stimulation
39 Hz Resonance
Spectral density stability constant
Re-synchronisation target without inducing seizure risk
Targeted Hub Gain
Strategic R_{\text{eff}} reduction
Focused bridge-node therapy — more efficient than whole-network approaches
🌍 The 5:1 Phase-Locking Hypothesis
RST v6.1 introduces an exploratory hypothesis regarding environmental coupling:
This 5:1 ratio constitutes a Perfect Fifth harmonic relationship across two octaves — one of the most stable coupling intervals in wave physics.
Hypothesis: During the 6.1-second ignition window, the brain acts as an open resonant system, utilising ambient ELF field harmonics as rhythmic scaffolding for cluster re-synchronisation.
Important caveat: The Schumann Resonance operates in the picoTesla range. RST proposes resonant coupling rather than amplitude-driven entrainment as the operative mechanism. This is explicitly framed as a hypothesis requiring dedicated experimental investigation.
Falsifiability criteria:
ELF-shielded sessions should produce measurable widening of the 6.1s window
Synthetic 7.83 Hz interference should disrupt Fiedler Vector node stability
RSN correlations should vary with geomagnetic intensity
Simultaneous EEG + geomagnetic co-registration required to confirm phase-lock
Preliminary observation: Sessions conducted in high-EMI environments (urban) showed mean ignition delay of 6.4s (±0.3s) vs 6.1s (±0.1s) in low-EMI environments (rural). Reported as anecdotal only — multiple confounds present.
📂 Repository Structure
RST-v6.1/
│
├── README.md                          # This file
│
├── Preprint/
│   └── RST_v6.1_Preprint_Botha_2026.pdf    # Full academic preprint
│
├── Toolkit/                           # Termux/Python mobile analysis tools
│   ├── rst_rsn_calculator.py          # Compute RSN from EEG spectral data
│   ├── rst_laplacian.py               # Algebraic connectivity + Fiedler Vector
│   ├── rst_batch_analysis.py          # Batch EEG file processing
│   └── README_toolkit.md              # Toolkit usage instructions
│
├── Logs/                              # Anonymised session records
│   ├── Subject_0284_sessions.md       # 500+ iteration log excerpts
│   └── CaseStudyAlpha_syncope.md      # Tussive syncope observations
│
├── TUH_Analysis/                      # N=581 corpus analysis
│   ├── summary_statistics.md          # Core findings and regression output
│   └── methodology_notes.md           # Analysis approach and parameters
│
└── LICENSE                            # Apache 2.0
Note: Toolkit scripts, session logs, and TUH analysis summaries are being prepared for upload. The preprint PDF contains the complete theoretical and empirical documentation in the interim.
🛠️ Getting Started (Termux Toolkit)
The RST analysis tools are designed for mobile accessibility — no institutional hardware required.
Prerequisites
pkg install python
pip install numpy scipy
Compute RSN from EEG data
# Example usage (full scripts in /Toolkit/)
from rst_rsn_calculator import compute_rsn

rsn = compute_rsn(
    eeg_signal=your_signal,
    sample_rate=256,
    method='welch'
)
print(f"RSN: {rsn:.3f}")
# RSN < 1: under-coupled | RSN = 1: critical | RSN > 1: chaotic
📋 Limitations
RST v6.1 explicitly acknowledges the following boundary conditions:
N=1 optimisation: Coupling dynamics most extensively characterised in Subject 0284. Cross-subject validation required.
Healthy brain baseline: RST has not yet been validated against non-injured EEG datasets. Normative RSN distributions are outstanding.
Causality gap: The 5:1 Schumann relationship is a correlation. No causal mechanism confirmed. ELF-shielding controls not yet performed.
Resource constraints: All analysis performed via Termux mobile tools. Institutional EEG hardware replication recommended.
Schumann amplitude: Schumann field strengths operate in the picoTesla range. Resonant coupling mechanism remains theoretical.
🔬 Future Work
Priority
Investigation
Expected Outcome
High
Healthy brain baseline (normative RSN)
Contextualise injured-brain findings
High
ELF shielding experiment
Test 5:1 phase-locking hypothesis
Medium
Subject 0285 longitudinal tracking
Second closed case validation
Medium
39 Hz tACS clinical pilot
Neuromodulation protocol development
Low
Wearable RSN monitor
Real-time coupling state feedback
Low
Cross-scale coupling (EEG + HRV + geomagnetic)
Multi-modal validation
🤝 Collaboration
RST is open-source and actively seeking:
Neuroscientists with EEG dataset access
Clinicians working in TBI rehabilitation
Engineers interested in wearable RSN monitoring
Independent researchers willing to replicate findings
Contact: jjbotha1991@gmail.com | +27 79 391 7254
📄 Citation
@misc{botha2026rst,
  author    = {Botha, JJ},
  title     = {Scale-Analogous Coupling in Neural Recovery: 
               A Topological Framework for Post-Axonal Shearing Stabilisation},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  note      = {Resonant Scaling Theory v6.1. Apache 2.0 Open Source.}
}
📜 License & Declaration
Licensed under the Apache License 2.0 — see LICENSE file for details.
This work was conducted independently without institutional affiliation or external funding. Subject 0284 and Case Study Alpha data collected with informed consent under anonymised subject coding. TUH-EEG Corpus accessed as a public research dataset under published terms of use.
The author acknowledges AI-assisted synthesis (DeepSeek, Claude/Anthropic) in the formulation of theoretical connections. All empirical analysis and core theoretical claims originate from the author's own research process.
"The line remains unbroken. Not because it was never broken. But because it learned to resonate at 39 Hz."
— JJ Botha, Resonant Keeper, Nelspruit 2026