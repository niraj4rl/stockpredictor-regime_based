import pandas as pd
from dateutil.relativedelta import relativedelta
from utils.config import TRAIN_YEARS, TEST_MONTHS, PURGE_GAP_DAYS


class WalkForwardSplitter:
    def __init__(
        self,
        train_years: int = TRAIN_YEARS,
        test_months: int = TEST_MONTHS,
        purge_gap_days: int = PURGE_GAP_DAYS,
    ):
        self.train_years = train_years
        self.test_months = test_months
        self.purge_gap_days = purge_gap_days

    def split(self, df: pd.DataFrame):
        df = df.sort_index()
        start = df.index[0]
        train_end = start + relativedelta(years=self.train_years)

        fold = 0
        while True:
            test_end = train_end + relativedelta(months=self.test_months)
            purge_start = train_end - relativedelta(days=self.purge_gap_days)
            train_mask = df.index < purge_start
            test_mask = (df.index >= train_end) & (df.index < test_end)

            if test_mask.sum() == 0:
                break

            train_df = df[train_mask]
            test_df = df[test_mask]

            if len(train_df) < 200:
                train_end = test_end
                continue

            yield fold, train_df, test_df, train_end, test_end
            fold += 1
            train_end = test_end

    def n_folds(self, df: pd.DataFrame) -> int:
        return sum(1 for _ in self.split(df))

    def describe(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for fold, train, test, te, tse in self.split(df):
            purge_start = te - relativedelta(days=self.purge_gap_days)
            rows.append({
                "fold": fold,
                "train_start": train.index[0].date(),
                "train_end": train.index[-1].date(),
                "purge_start": purge_start.date(),
                "purge_gap_days": self.purge_gap_days,
                "train_rows": len(train),
                "test_start": test.index[0].date(),
                "test_end": test.index[-1].date(),
                "test_rows": len(test),
            })
        return pd.DataFrame(rows)
