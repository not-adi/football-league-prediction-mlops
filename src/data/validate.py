import logging, pandas as pd
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    pass

@dataclass
class ValidationReport:
    passed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    @property
    def is_valid(self):
        return len(self.failures) == 0

    def summary(self):
        lines = [f'Passed: {len(self.passed)}, Warnings: {len(self.warnings)}, Failures: {len(self.failures)}']
        for f in self.failures: lines.append(f'  FAIL: {f}')
        return '\n'.join(lines)

def validate_raw_matches(df, league_code):
    report = ValidationReport()
    if df.empty:
        report.failures.append('DataFrame is empty')
    else:
        report.passed.append(f'{len(df)} rows loaded')
    finished = df[df['status'] == 'FINISHED']
    null_scores = finished[['home_goals','away_goals']].isnull().any(axis=1).sum()
    if null_scores > 0:
        report.failures.append(f'{null_scores} FINISHED matches have null scores')
    else:
        report.passed.append('No null scores in finished matches')
    dupe_ids = df['match_id'].duplicated().sum()
    if dupe_ids > 0:
        report.failures.append(f'{dupe_ids} duplicate match IDs')
    else:
        report.passed.append('No duplicate match IDs')
    logger.info(f'Validation [{league_code}]: {report.summary()}')
    if not report.is_valid:
        raise DataValidationError(f'Validation failed for {league_code}:\n{report.summary()}')
    return report
