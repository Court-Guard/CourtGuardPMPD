# Policy Workspace

Place the source policy document used for bootstrap in this directory.

Typical workflow:

1. add the policy file here
2. run `python evaluate.py --mode pmpd ...` from `CourtGuard/`
3. allow the bootstrap pipeline to build the markdown tree and PMPD store

The anonymous submission bundle intentionally excludes provider-specific local policy files and keeps only the PMPD stores used in the reported experiments.
