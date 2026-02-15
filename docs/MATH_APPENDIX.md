# Appendix A: Formal Derivation of the Square-Root Scaling Law

RST v6.2.1 defines the recovery interval ($T_c$) through the interaction of the spectral gap and the metabolic coherence envelope.

### 1. The Reduced Envelope Equation
Given the linearized damping of the signal amplitude ($r$) coupled with energy accumulation ($E$):
$$\dot{r} = -\lambda_2 E r$$
$$\dot{E} \approx \beta$$

### 2. Integration of the Gaussian Decay
By integrating the early-phase dynamics where $E(t) \approx \beta t$:
$$\int \frac{dr}{r} = -\lambda_2 \beta \int t \, dt$$
$$\ln(r) = -\frac{1}{2} \lambda_2 \beta t^2 + C$$

Solving for the recovery threshold ($\epsilon$):
$$T_{recovery} = \sqrt{\frac{2 \ln(1/\epsilon)}{\lambda_2 \beta}}$$

**Conclusion:** This identifies $T_c \propto \lambda_2^{-1/2}$ as the fundamental scaling limit for post-traumatic resynchronization.
