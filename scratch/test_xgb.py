import joblib
import numpy as np

model = joblib.load('backend/models/arabic_dialect_xgboost_model.pkl')
print(type(model))
if hasattr(model, 'feature_importances_'):
    print("Has feature importances! len:", len(model.feature_importances_))
else:
    print("No feature importances.")
