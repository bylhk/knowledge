# Reporting
Reporting is important in MLOps, we need to monitor the data and model drift.
The efficiency and time lag are the critical parts of reporting.

## Multi-level aggregation
If the models serve millions of transactions, the optimisation become important in analysis and reporting. Mid layers can avoid the repeated efforts in measurement.<br>

For example, the transactions can be clustered by different categories and models. The users need different combinations for analysis; the reports also contain different metrics.<br>
The metrics (equations) can be broken down into some variables (mid-layers) by categories. Then we can reuse the mid-layers for the final metrics according to the users' needs.

## Instant report
When we have multiple layers, the dashboards don't measure metrics from the raw data; the instant report becomes possible.
