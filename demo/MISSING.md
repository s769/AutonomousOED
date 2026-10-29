# Metrics and products not in the talk UI

- **Displacement-UQ rasters for feasible-remainder networks.** Repaired-network
  frames were generated on job 60127421 (same adjoint pipeline, rebuilt `K`,
  `r_sq=1e4`). Remainder views still hide the overlay rather than showing a
  mismatched field.
- **Optimality certificate for the repairs.** The original-network ~87% bound
  does not apply to this constrained completion class (feasible 175-sensor
  networks that contain the retained sensors). No new certificate was computed.
- **Independent GPU Cholesky of the three *f(S)* values: completed.** Incremental
  Cholesky on 1× A100 80GB (job 60127421) reproduced the checkpoint *f(S)* values
  to fp32 noise (~0.2 nats on a ~2.87×10⁵ scale). `objectives_verified.json`
  records the method. Dense slogdet of the 73,500×73,500 `K_S` is not used.
