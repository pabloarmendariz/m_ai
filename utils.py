import joblib

def saving_model(model, name):
    joblib.dump(model, f"models/{name}.joblib")
    print("Model saved.")