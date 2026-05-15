import joblib
import numpy as np
import shap

model = joblib.load('backend/models/arabic_dialect_xgboost_model.pkl')
print(model.steps)

# The model is a pipeline. usually the final step is the classifier
clf_name, clf = model.steps[-1]
print("Classifier type:", type(clf))
if hasattr(clf, 'feature_importances_'):
    print("Has feature importances! len:", len(clf.feature_importances_))

# Let's generate a dummy input that matches the scaler
dummy_input = np.random.randn(1, 162) # 162 features for the dialect_classifier.py
try:
    transformed_input = dummy_input
    for name, step in model.steps[:-1]:
        transformed_input = step.transform(transformed_input)
        
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(transformed_input)
    print("SHAP values shape:", np.array(shap_values).shape)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("SHAP failed:", e)
