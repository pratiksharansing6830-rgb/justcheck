# Adaptive Fraud Strategy Validation

A Python prototype for evaluating fraud-detection strategies on historical data before simulating their use on incoming transactions.

## Current scope

Phase 1 provides CSV loading and dataset inspection only. No transaction fields, labels, entity identifiers, or feature availability are assumed. Inspect the candidate dataset before defining features or model inputs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest
```

## Dataset inspection

Use `fraud_validation.data.loader.load_csv` to read a CSV and `fraud_validation.data.validation.validate_dataframe` to inspect its columns, types, missing values, and duplicate rows. Pass `required_columns` only after deciding which fields the selected dataset and first experiment actually require. The validation report surfaces diagnostics; it does not clean data or infer fraud labels.