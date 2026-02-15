### 2. The Recovery Scaling Law: Refinement (v6.2.1)
Original iterations of RST hypothesized a quadratic scaling law ($1/\lambda^2$). However, rigorous analytical derivation and simulation of the coupled envelope system ($\dot{r} = -\lambda_2 E r$) reveal that the recovery time follows a **Square-Root Scaling Law**:

$$T_{recovery} \propto \frac{1}{\sqrt{\lambda_2}}$$

#### Derivation Summary:
In the early phase of neural recovery, where energy ($E$) accumulates linearly ($E \approx \beta t$), the signal amplitude ($r$) follows a Gaussian decay:
$$r(t) = \exp\left(-\frac{1}{2} \lambda_2 \beta t^2\right)$$
Solving for the recovery threshold yields the $1/\sqrt{\lambda_2}$ relationship. This identifies the **Square-Root** as the fundamental limit of adaptive damping in partitioned oscillators.


