# SC26 presenter script (~2 minutes)

**Line.** Expensive physics calculations can be reused to adapt a high-fidelity
observing system when deployment constraints change.

1. **Physics is already done.** “We computed the wave-propagation operators
   once and stored them as `K`. Design only reads that matrix.” Click
   **Compute 175** to run unconstrained greedy D-opt live on the held
   allocation (or **Load K** then **Start selection** for a timed reveal).
   Sensors appear on the map as they are chosen. The color field is
   posterior σ of seafloor displacement (metres) — a diagnostic. The design
   criterion is D-optimal information gain about the inferred parameters.
   **Playback** still animates a precomputed reference if you need a dry run.
2. **Ask the audience.** North or south cable corridor unavailable (a survey
   constraint, not an earthquake, not sequential inference from data already
   collected). Read the one-liner:
   “South branch unavailable: 28 selected locations must be replaced.”
3. **Repair the network.** Gray = kept, teal = replacements, × = lost.
   “Keep every feasible sensor; greedy-fill back to 175.” Either click
   **Repair network** for a one-shot run, or **Load & factorize** beforehand
   and **Start selection** when you want the greedy loop to start.
4. **Compare remainder vs repaired.** Replacements recover Δf nats of
   information gain. The repaired network is about 99% of the original greedy
   *f(S)* even though that original design is infeasible under the exclusion.

Reset demo between takes. Full screen for the projector.
