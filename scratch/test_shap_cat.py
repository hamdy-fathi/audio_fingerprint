import joblib
import numpy as np
import shap
import json

def test_shap_category_heatmap():
    model = joblib.load('backend/models/arabic_dialect_xgboost_model.pkl')
    clf = model.steps[-1][1]
    
    # Dummy input
    np.random.seed(42)
    dummy_input = np.random.randn(1, 162)
    transformed_input = dummy_input
    for name, step in model.steps[:-1]:
        transformed_input = step.transform(transformed_input)
        
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(transformed_input) # (1, 162, 4)
    
    categories = {
        'Vocal Tract (MFCC)': list(range(0, 40)),
        'Speech Dynamics (Delta)': list(range(40, 120)),
        'Tonal Profile (Chroma)': list(range(120, 144)),
        'Spectral Shape & Contrast': list(range(144, 158)) + list(range(158, 162))
    }
    
    dialects = ['Egyptian Arabic', 'Gulf Arabic', 'Levantine Arabic', 'Maghrebi Arabic']
    
    heatmap_data = []
    
    for cat_name, indices in categories.items():
        cat_impacts = []
        for d_idx in range(4):
            # Sum of SHAP values for this category and this dialect
            impact = np.sum(shap_values[0, indices, d_idx])
            cat_impacts.append(impact)
            
        # Softmax to get percentages
        exp_impacts = np.exp(cat_impacts - np.max(cat_impacts)) # subtract max for numerical stability
        softmax = exp_impacts / np.sum(exp_impacts)
        percentages = [round(float(p) * 100) for p in softmax]
        heatmap_data.append(percentages)
        
        print(f"Category: {cat_name}")
        print(f"Impacts: {cat_impacts}")
        print(f"Percentages: {percentages}\n")

if __name__ == "__main__":
    test_shap_category_heatmap()
