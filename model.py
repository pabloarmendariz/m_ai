import json
from preprocessing import create_pipeline, create_preprocessing_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

features = json.load(open("config.json", "r"))
date_features = features["date_features"]
numerical_features = features['numerical_features']
categorical_features = features['categorical_features']
param_grid = features['param_grid']

preprocessor = create_preprocessing_pipeline(
    date_feat=date_features,
    num_feat=numerical_features,
    cat_feat=categorical_features
)

estimator = create_pipeline(
    processor=preprocessor,
    model=GradientBoostingRegressor()
)

model = GridSearchCV(
    estimator=estimator,
    param_grid=param_grid,
    cv=4,
    n_jobs=-1
)

if __name__=="__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from utils import saving_model
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(filepath_or_buffer="dataset/stores_sales_forecasting.csv", encoding="latin-1")
    df['profit_ratio'] = df["Profit"]/df["Sales"]
    X = df.drop(columns="profit_ratio")
    y = df["profit_ratio"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
    model.fit(X_train, y_train)
    saving_model(model, "validation_gb_model")
    pred = model.best_estimator_.predict(X_test)
    plt.scatter(y_test, pred)
    plt.show()
    