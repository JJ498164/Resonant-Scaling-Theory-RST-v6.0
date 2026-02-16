# Privacy & Security in Federated RST

## 1. Data Minimization (ZKT)
The RST Bridge utilizes **Zero-Knowledge Topology**. 
* **Captured Locally:** Raw spikes, somatic phase, and intent.
* **Federated Globally:** Anonymized spectral gap ($\lambda_2$) and persistence decay ($T$).

## 2. Secure Edge Execution
All **Stiffening Injections** are computed and executed on-device. No external entity can trigger a 39.1 Hz pulse; the N1 firmware validates all rst_stiffening_protocol commands against local metabolic safety bounds (the 6.1s invariant).

## 3. Global Calibration
Federated data is used solely to tune the **Scaling Law** constant ($K$). 
Current Multi-Implant Baseline: 
* **$\lambda_2$ Stability:** 0.74
* **Target $T$:** ~3.9s
