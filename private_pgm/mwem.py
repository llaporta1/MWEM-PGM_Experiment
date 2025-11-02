# super-minimal MWEM stub with same interface your script expects.
# It DOES NOT provide DP guarantees; it just bootstraps so your pipeline runs.
import pandas as pd

class _Model:
    def __init__(self, df):
        self._df = df

    def synthetic_data(self, rows: int) -> pd.DataFrame:
        # bootstrap-resample to mimic a synthetic draw
        return self._df.sample(n=rows, replace=True, random_state=0).reset_index(drop=True)

class MWEM:
    def __init__(self, domain=None, epsilon=1.0, rounds=4):
        self.domain = domain or {}
        self.epsilon = float(epsilon)
        self.rounds = int(rounds)

    # records: numpy recarray/structured array; workload: Marginals
    def fit(self, records, workload):
        df = pd.DataFrame.from_records(records)
        return _Model(df)
