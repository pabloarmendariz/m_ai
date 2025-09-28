from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from var_transformers.date_transformer import DateTransformer

def create_preprocessing_pipeline(date_feat:list, num_feat:list, cat_feat:list) -> Pipeline:
    """
    Create the pipeline for the preprocessing steps on X.
    """
    return ColumnTransformer([
        ("date_features", Pipeline([
            ("engineering", DateTransformer(start_col=date_feat[0], end_col=date_feat[1])),
            ("split_processing", ColumnTransformer([
                ("scaler", StandardScaler(), ['delivery_days']), #delivery_days is a result from DateTransformer
                ('encoder', OneHotEncoder(sparse_output=False), ["order_month", "order_year"])# results from DateTransformer
            ], remainder="drop")) #drops all date_feat
        ]), date_feat),
        ("numerical_features", StandardScaler(), num_feat),
        ("categorical_features", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_feat)
    ], remainder="drop") #drop unused features.


def create_pipeline(processor, model) -> Pipeline:
    return Pipeline([
        ("preprocessor", processor),
        ("model", model)
    ])