"""
Note: This file was mainly used for exploratory purposes only.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv(filepath_or_buffer="dataset/stores_sales_forecasting.csv", encoding="latin-1")

print(df.head(n=10))

"""
We have two date columns. Some information from the order date might be useful later such as month and year, but the day is a detail that will not bring much value to a ML model.
I will also create a variable that calculates the differences between dates.
Note: No missing variable in this dataset, so no imputation needed.
"""
df.info()

def date_diff(data:pd.DataFrame, start:pd.Series, end:pd.Series) -> pd.Series:
    """
    Formats and calculates the differences between end date and start date.
    Returns date difference in days as an int.
    """
    ddiff = pd.to_datetime(data[end], format="%m/%d/%Y") - pd.to_datetime(data[start], format="%m/%d/%Y")
    return ddiff.dt.days

date_features = ["Order Date", "Ship Date"] # Order is quite important (start, end)
class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, start_col:str, end_col:str) -> None:
        self.start_col = start_col
        self.end_col = end_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X) -> pd.DataFrame:
        start_dates = pd.to_datetime(X[self.start_col], format="%m/%d/%Y")
        end_dates = pd.to_datetime(X[self.end_col], format="%m/%d/%Y")

        result = pd.DataFrame({
            'delivery_days': (end_dates - start_dates).dt.days,
            'order_month': start_dates.dt.month,
            'order_year': start_dates.dt.year
        })
        
        return result

"""
Quickly I realise ROW ID and Postal Code have a uniform distribution. First, it means ROW ID doesn't have any added value to our exercice and needs to be dropped.
Second, Postal Code is not well formated and should be encoded and maybe dropped.
Let's see if the unique values of all columns, including Postal Code, correspond to 90% of the overall dataset.
If it does, then it means, the feature isn't giving much information and need to be dropped.
The same can be said if there is a constant. We will check for that as well.
I also want to excluse Customer Name because we already have a Customer ID variable and both would negate their impact.
Same for Product Name.
"""
sns.pairplot(data=df)
plt.show()

def col_to_drop(data:pd.DataFrame) -> list:
    data_size = data.shape[0]
    def conditions(col):
        col_size = len(data[col].drop_duplicates())
        return col_size/data_size > 0.9 or col_size == 1
    cols = [col for col in data.columns if conditions(col)]
    return cols

columns_to_drop = col_to_drop(df)
columns_to_drop.insert(0, "Order Date")
columns_to_drop.insert(0, "Ship Date")

"""
Before dropping the Customer and Product Names, I wanted to make sure it was the right decision.
"""
cust_id_size = len(df['Customer ID'].drop_duplicates())
cust_name_size = len(df['Customer Name'].drop_duplicates())
print(cust_id_size == cust_name_size)

prod_id_size = len(df['Product ID'].drop_duplicates())
prod_name_size = len(df["Product Name"].drop_duplicates())
print(prod_id_size == prod_name_size)

columns_to_drop.insert(0, "Customer Name")
columns_to_drop.insert(0, "Product ID")

"""
With more explanation, I think I could have used Order ID. This variable could change the layout of this dataset to a wide format.
Time doesn't allow us for such a change in structure so I will drop it.
"""
columns_to_drop.insert(0, "Order ID")

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop:list):
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X:pd.DataFrame):
        return X.drop(columns=self.columns_to_drop, errors="ignore")

"""
After creating my function, I realise Postal Code is not amongst the columns_to_drop. Therefore, it needs a new formating because it is in a int format.
"""
categorical_features = [col for col in df.columns if df[col].dtypes == object and col not in columns_to_drop]
categorical_features.insert(0,"Postal Code")
print(f"Categorical Features: {categorical_features}")
"""
Amongst the numerical variables, two (2) of them have similar patterns and can be both predicted.
Sales and profit represent two variable important to any retailer. Unfortunately, high sales doesn't equal high profit as seen in the first graph.
Therefore, I believe our target of understanding prediction should be un the ratio of profit by sales.
This will reduce the redundancy in similar variables and give more weight to positive variables.
The graph below start to show more relationship amongst numerical variables than before.
Therefore we can add Sales and Profit to columns_to_drop and remain with profit_ratio.
"""
df['profit_ratio'] = df["Profit"]/df['Sales']
df['delivery_days'] = date_diff(df, "Order Date", "Ship Date")
sns.pairplot(df, y_vars="profit_ratio", x_vars=["Quantity", "Discount", "delivery_days"])
plt.show()

df.drop(columns='delivery_days') # This is included in the pipeline and was for visual purpose only.
columns_to_drop.insert(0, "Sales")
columns_to_drop.insert(0, "Profit")

numerical_features = ["Quantity", "Discount"]

"""
All variables are now ready. We need to create the pipeline.
"""
X = df.drop(columns='profit_ratio')
y = df["profit_ratio"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

preprocessor = ColumnTransformer([
    ("date_features", Pipeline([
        ("engineering", DateTransformer(start_col=date_features[0], end_col=date_features[1])),
        ("split_processing", ColumnTransformer([
            ("scaler", StandardScaler(), ['delivery_days']), #delivery_days is a result from DateTransformer
            ('encoder', OneHotEncoder(sparse_output=False), ["order_month", "order_year"])# result from DateTransformer
        ], remainder="drop"))
    ]), date_features),
    ("numerical_features", StandardScaler(), numerical_features),
    ("categorical_features", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features)
], remainder="drop")


param_grid_gb = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 4, 5]
    #'model__min_samples_split': [2, 5],
    #'model__subsample': [0.8, 1.0]
}

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ('model', GradientBoostingRegressor())
])

model = GridSearchCV(
    estimator=pipeline,
    cv = 4,
    param_grid=param_grid_gb,
    n_jobs=-1
)

model.fit(X_train, y_train)
joblib.dump(model, "models/profit_ratio_gb_model.joblib")
pred = model.best_estimator_.predict(X_test)
