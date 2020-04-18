import joblib

loaded_model = joblib.load('finalized_dt_model.sav')
result = loaded_model.score(X_test, y_test)
print(result)