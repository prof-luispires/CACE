CMARL Expert v7.5 - Selective Reliability Modulation

Design fixed before test execution:
- Temporal split: 50% train / 20% validation / 30% independent test.
- Expert thresholds learned on training only.
- Expert reliability estimated on validation only.
- Test set is never used for tuning.
- v7.5 selectively modulates only the Drift Expert, identified in v7.3 diagnostics as the problematic expert.
- Fixed alpha = 0.60:
    c'_drift(t) = c_drift(t) * [alpha + (1-alpha) * r_drift]
- Other expert confidence scores are unchanged.
- Logistic coordinator is trained on the modulated expert scores.
- Coordinator threshold is selected on validation only.
- Hybrid gate remains the v7.3 rule: at least two expert confidence scores >= 0.65.
- Five seeds: 42, 123, 321, 777, 999.
- Nine sensors.
- Low, Medium, High anomaly-load scenarios.
- Paired Wilcoxon tests: 45 sensor/seed pairs per scenario.

The package also includes the recomputed v7.3 hybrid reference under exactly the same data splits,
allowing a direct v7.5 vs v7.3 comparison.
