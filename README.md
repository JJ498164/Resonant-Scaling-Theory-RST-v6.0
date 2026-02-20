# Resonant Scaling Theory (RST)  
### A Two-Timescale Dynamical Model of Thalamocortical Resonance and Metabolic Constraint

---

## 1. Overview

Resonant Scaling Theory (RST) proposes a minimal multiscale dynamical model describing the interaction between:

- A fast thalamocortical oscillatory subsystem (~39 Hz attractor)
- A slow metabolic recovery envelope (~6.1 s time constant)
- Gain coupling mediated by energetic constraints

The framework formalizes how oscillatory stability is regulated by metabolic dissipation across distinct temporal scales.

RST does **not** claim universal physical applicability.  
It is a dynamical systems model targeting neural resonance stability under metabolic constraint.

---

## 2. Core Hypothesis

Neural gamma stability is not determined solely by instantaneous power, but by coupling between:

- Fast oscillatory dynamics (τ_f ≈ 12.8 ms delay loop)
- Slow recovery dynamics (τ_s ≈ 6.1 s envelope)

Instability emerges when energetic injection exceeds dissipation capacity, characterized by a dimensionless control parameter.

---

## 3. Mathematical Formulation (Minimal Model)

### State Variables
- x(t): Fast oscillatory neural state
- E(t): Slow metabolic envelope
- G(E): Gain modulation term

### Governing Equations

Fast dynamics:

dx/dt = F(x, τ_f) − G(E)x

Slow envelope:

τ_s dE/dt = −E + H(x)

Where:
- F(x) supports a limit cycle near 39 Hz
- H(x) scales with oscillatory energy
- τ_f ≈ 12.8 ms
- τ_s ≈ 6.1 s

Linearization yields a coupled eigenvalue system governing resonance stability.

---

## 4. Neural Reynolds Number (Control Parameter)

RST introduces a dimensionless stability parameter:

𝒩 = P_osc / (γE)

Where:
- P_osc = oscillatory power injection
- γE = metabolic dissipation rate

Interpretation:
- 𝒩 < 1: Stable regime
- 𝒩 > 1: Instability / entropy escalation

This parameter functions analogously to classical Reynolds-type instability thresholds in dynamical systems.

---

## 5. Testable Predictions

RST generates falsifiable predictions:

1. Metabolic modulation alters stability bandwidth without shifting peak gamma frequency.
2. Phase–Amplitude Coupling between 8.75 Hz phase and ~39 Hz amplitude collapses when 𝒩 > 1.
3. Entropy measured at ~6 s windows predicts instability earlier than absolute power metrics.

All predictions are empirically testable using EEG/MEG datasets.

---

## 6. Repository Structure

- /theory – Mathematical derivations
- /simulation – Minimal coupled oscillator implementation
- /validation – EEG processing pipeline (MNE / Tensorpac compatible)
- /analysis – Stability and entropy metrics
- /predictions – Experimental test framework

---

## 7. Scope and Limitations

RST:
- Is a dynamical systems model of neural resonance
- Does not claim to supersede established neural mass models
- Requires empirical validation across independent datasets
- Remains under active formal development

---

## 8. Citation

Author: JJ498164  
Title: Resonant Scaling Theory (RST)  
Version: 6.0  
Year: 2026  
Repository: https://github.com/JJ498164/Resonant-Scaling-Theory-RST-v6.0