"""Dataset-agnostic structural and quality diagnostics."""

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    row_count: int
    columns: tuple[str, ...]
    dtypes: tuple[tuple[str, str], ...]
    missing_required_columns: tuple[str, ...]
    duplicate_columns: tuple[str, ...]
    null_counts: tuple[tuple[str, int], ...]
    duplicate_rows: int

    @property
    def is_valid(self) -> bool:
        """Whether the frame is nonempty and meets its requested schema."""
        return (
            self.row_count > 0
            and not self.missing_required_columns
            and not self.duplicate_columns
        )


def validate_dataframe(
    frame: pd.DataFrame,
    required_columns: Iterable[str] = (),
) -> ValidationReport:
    """Summarize dataset structure without assuming fraud-specific fields."""
    columns = tuple(str(column) for column in frame.columns)
    column_set = set(columns)
    required = tuple(dict.fromkeys(required_columns))
    missing_required = tuple(column for column in required if column not in column_set)

    seen: set[str] = set()
    duplicate_columns = []
    for column in columns:
        if column in seen and column not in duplicate_columns:
            duplicate_columns.append(column)
        seen.add(column)

    dtypes = tuple(
        (str(column), str(frame.dtypes.iloc[index]))
        for index, column in enumerate(frame.columns)
    )
    null_counts = tuple(
        (str(frame.columns[index]), int(frame.iloc[:, index].isna().sum()))
        for index in range(frame.shape[1])
    )

    return ValidationReport(
        row_count=len(frame),
        columns=columns,
        dtypes=dtypes,
        missing_required_columns=missing_required,
        duplicate_columns=tuple(duplicate_columns),
        null_counts=null_counts,
        duplicate_rows=int(frame.duplicated().sum()),
    )