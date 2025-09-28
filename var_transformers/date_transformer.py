import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateTransformer(BaseEstimator, TransformerMixin):
    """
    Class that will handle transformation from start and end dates.
    First, it will create a duration variable.
    Second, it will create a month variable.
    Third, it will create a year variable.
    """
    def __init__(self, start_col:str, end_col:str) -> None:
        self.start_col = start_col
        self.end_col = end_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        start_dates = pd.to_datetime(X[self.start_col], format="%m/%d/%Y")
        end_dates = pd.to_datetime(X[self.end_col], format="%m/%d/%Y")

        result = pd.DataFrame({
            'delivery_days': (end_dates - start_dates).dt.days,
            'order_month': start_dates.dt.month,
            'order_year': start_dates.dt.year
        })
        return result