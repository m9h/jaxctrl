---
category: research
section: results
weight: 30
title: "Results"
status: draft
---

# Results

This section is populated by the autoresearch loop defined in
`autoresearch/program.md`. Each experiment appends one tab-separated row to
`autoresearch/results.tsv` with the columns

```
commit  metric_value  parameters_json  status  description
```

The headline metric is

$$\textrm{control\_advantage} = \frac{\textrm{cost}_{\textrm{classical}} - \textrm{cost}_{\textrm{jaxctrl}}}{\textrm{cost}_{\textrm{classical}}}$$

evaluated at matched wall time, where `cost` is the LQR value function
$J = \mathrm{tr}(P\,\Sigma_0)$ with $\Sigma_0 = I$.

The current baseline (linearised Van der Pol around the origin, $n=2$,
$\mu=1$) yields a control advantage of order $10^{-15}$ — both solvers agree
to floating-point precision, which is the expected starting point. The
autoresearch agent will explore non-trivial regimes: stiff systems, very
high state dimensions, parameterised cost matrices that exercise the
gradient path, and hypergraph driver-node placements where the differentiable
formulation is expected to dominate.

A summary table will be regenerated from `results.tsv` for each manuscript
build; the column schema mirrors the file format above.
