import pandas as pd
import joblib
import seaborn as sns
from sklearn.model_selection import train_test_split

def saving_model(model, name) -> None:
    joblib.dump(model, f"models/{name}.joblib")
    print("Model saved.")

def get_data() -> pd.DataFrame:
    return pd.read_csv(filepath_or_buffer="dataset/stores_sales_forecasting.csv", encoding="latin-1")

def summary_graph(data):
    sns.pairplot(data=data[["Profit", "Sales", "Quantity", "Discount"]])

def create_profit_ratio(data) -> pd.Series:
    return data["Profit"]/data["Sales"]

def ratio_graph(data):
    sns.pairplot(data=data, y_vars="profit_ratio", x_vars=["Quantity", "Discount"])

def create_X_y(data:pd.DataFrame):
    X = data.drop(columns="profit_ratio")
    y = data["profit_ratio"]
    return train_test_split(X, y, test_size=0.2)

def load_model(name):
    joblib.load(f"models/{name}.joblib")