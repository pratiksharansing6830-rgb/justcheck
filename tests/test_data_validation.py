import pandas as pd

from fraud_validation.data.loader import load_csv
from fraud_validation.data.validation import validate_dataframe


def test_load_csv_reads_input_without_schema_assumptions(tmp_path):
    csv_path = tmp_path / "transactions.csv"
    csv_path.write_text("recorded_at,value\n2025-01-01,12.5\n")

    frame = load_csv(csv_path)

    assert list(frame.columns) == ["recorded_at", "value"]
    assert frame.loc[0, "value"] == 12.5


def test_validation_reports_schema_and_quality_diagnostics():
    frame = pd.DataFrame({"event_time": ["t1", "t1"], "amount": [None, 10]})

    report = validate_dataframe(frame, required_columns=("event_time", "amount"))

    assert report.is_valid
    assert report.columns == ("event_time", "amount")
    assert report.null_counts == (("event_time", 0), ("amount", 1))
    assert report.duplicate_rows == 0
    assert len(report.dtypes) == 2


def test_validation_marks_missing_required_columns_invalid():
    report = validate_dataframe(pd.DataFrame({"value": [1]}), ("label",))

    assert not report.is_valid
    assert report.missing_required_columns == ("label",)


def test_validation_marks_empty_data_invalid_and_reports_duplicates():
    frame = pd.DataFrame({"value": [1, 1]})

    report = validate_dataframe(frame.iloc[:0])
    duplicate_report = validate_dataframe(frame)

    assert not report.is_valid
    assert duplicate_report.duplicate_rows == 1