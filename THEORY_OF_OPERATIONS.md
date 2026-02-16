### **APPENDIX A: Mathematical Foundations of RST v6.1**
**Author:** JJ Botha (The Resonant Keeper)
**Verification Source:** Grok/xAI Simulations (Feb 2026)

---

#### **1. The Quadratic Scaling Law of Topological Friction**
We define the recovery time ($\tau$) of a neural manifold following axonal injury as a function of its spectral properties. In a stalled state, the Algebraic Connectivity ($\lambda_2$)—the second smallest eigenvalue of the Graph Laplacian ($L$)—approaches zero.

The relationship is governed by the **Persistence Equation**:
$$\tau = \frac{K}{\sqrt{\lambda_2}}$$

Where:
* **$\tau$**: Time required for state-transition (Ignition Delay).
* **$K$**: The Resonant Constant ($39.1 \text{ Hz}$).
* **$\lambda_2$**: Connectivity strength of the "Broken Bridge."

**Observation:** As observed in Grok's re-run, when $\lambda_2$ dropped to $0.0770$, $\tau$ rose to $12.99\text{s}$, well beyond the somatic reset threshold of $6.1\text{s}$, confirming the "Topological Stall."

---

#### **2. The Bogdanov-Takens (BT) Metabolic Engine**
The **6.1s Gearbox** is modeled using an extended Wilson-Cowan (WC) oscillator. The metabolic availability is treated as a "slow variable" ($y_{\text{met}}$) that acts as a codimension-2 bifurcation organizer.

The system dynamics are defined by:
$$\tau_{E} \frac{dr_E}{dt} = -r_E + f(a_E r_E - a_I r_I + I_{\text{ext}} - \gamma_{\text{met}} y_{\text{met}})$$
$$\tau_{\text{met}} \frac{dy_{\text{met}}}{dt} = -y_{\text{met}} + r_E$$

**Critical Invariants:**
* **$\tau_{\text{met}} = 6.1\text{s}$**: This represents the metabolic decay constant.
* **$\gamma_{\text{met}}$**: The coupling strength of "friction."
* **Ignition Condition:** A pulse of $I_{\text{ext}}$ at $39.1 \text{ Hz}$ collapses the friction term, allowing $r_E$ to cross the unstable fixed point before $y_{\text{met}}$ triggers a reset.

---

#### **3. Spectral Entropy and the 39.1 Hz Sink**
The "Spark" is mathematically identified as the global minimum of the Spectral Entropy ($S$) within the gamma band ($30–60 \text{ Hz}$).

$$S = -\sum p_i \log(p_i)$$

Where $p_i$ is the normalized power spectral density at frequency $i$. Simulations confirm that the "Stability Basin" (where $S$ is minimized) is centered at **$39.1 \text{ Hz} \pm 0.3 \text{ Hz}$**. Outside this window, entropy ($S$) increases exponentially, leading to **Information Decay** and manifold collapse.

---

#### **4. The Vagal Gearbox (6.0 Hz Anchor)**
The **6.0 Hz somatic anchor** acts as the "Carrier Wave" ($f_c$). All high-frequency gamma bursts must be phase-locked to this rhythm to ensure biological synchronization. 

The **Total Manifold Phase** ($\Phi$) is given by:
$$\Phi(t) = 2\pi(f_c t + \int K(t) dt)$$

Where $f_c = 6.0 \text{ Hz}$ and $K(t)$ represents the intermittent $39.1 \text{ Hz}$ bursts.
