import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
import base64

def get_162_feature_names():
    names = []
    for f_name, size in [("MFCC", 20), ("Delta", 20), ("Delta²", 20), ("Chroma", 12), ("Contrast", 7), ("ZCR", 1), ("RMS", 1)]:
        for i in range(size):
            names.append(f"{f_name} {i+1} Mean")
        for i in range(size):
            names.append(f"{f_name} {i+1} Std")
    return names

COLORS = {
    'bg': '#120E0B',
    'surface': '#1A1410',
    'border': '#332822',
    'text': '#F5E6D3',
    'text_muted': '#A39688',
    'accent': '#D4A373'
}

DIALECT_COLORS = {
    'egyptian': '#EF4444',
    'gulf': '#3B82F6',
    'levantine': '#10B981',
    'maghrebi': '#F59E0B'
}

DIALECT_LABELS = {
    'egyptian': 'Egyptian (Egypt, Sudan)',
    'gulf': 'Gulf (Saudi, UAE, Qatar)',
    'levantine': 'Levantine (Lebanon, Syria)',
    'maghrebi': 'Maghrebi (Morocco, Algeria)'
}

def create_combined_plot():
    model = joblib.load('backend/models/arabic_dialect_xgboost_model.pkl')
    clf = model.steps[-1][1]
    
    # Dummy input
    dummy_input = np.random.randn(1, 162)
    transformed_input = dummy_input
    for name, step in model.steps[:-1]:
        transformed_input = step.transform(transformed_input)
        
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(transformed_input) # shape: (1, 162, 4)
    
    # Let's say predicted class is 0 (Egyptian)
    pred_class = 0
    shap_for_class = shap_values[0, :, pred_class]
    
    # Top 5 SHAP
    top_5_idx = np.argsort(np.abs(shap_for_class))[-5:]
    feature_names = get_162_feature_names()
    top_5_names = [feature_names[i] for i in top_5_idx]
    top_5_vals = shap_for_class[top_5_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS['bg'])
    
    # --- PLOT 1: SHAP ---
    ax1.set_facecolor(COLORS['bg'])
    y_pos = np.arange(len(top_5_names))
    colors = ['#4ADE80' if v > 0 else '#EF4444' for v in top_5_vals]
    ax1.barh(y_pos, top_5_vals, color=colors, height=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_5_names, color=COLORS['text'], fontsize=11, fontweight='bold')
    ax1.tick_params(colors=COLORS['text'])
    ax1.set_title("Top 5 Features Driving Prediction", color=COLORS['text'], fontsize=14, fontweight='bold', pad=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_color(COLORS['border'])
    
    # --- PLOT 2: XGBoost Weighted Category Match ---
    # We will compute a weighted distance. For this mock, just random percentages
    categories = ['Vocal Tract', 'Speech Dynamics', 'Spectral Shape', 'Tonal Profile', 'Band Contrast']
    dialects = list(DIALECT_COLORS.keys())
    data = np.random.randint(20, 100, size=(5, 4))
    
    ax2.set_facecolor(COLORS['surface'])
    im = ax2.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    for i in range(len(categories)):
        best_j = np.argmax(data[i])
        for j in range(len(dialects)):
            val = data[i, j]
            weight = 'bold' if j == best_j else 'normal'
            txt_color = 'white' if val < 50 else 'black'
            ax2.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    color=txt_color, fontsize=12, fontweight=weight)
                    
    ax2.set_xticks(range(len(dialects)))
    ax2.set_xticklabels([DIALECT_LABELS[d].split(' (')[0] for d in dialects],
                       color=COLORS['text'], fontsize=11, fontweight='bold')
    ax2.set_yticks(range(len(categories)))
    ax2.set_yticklabels(categories, color=COLORS['text'], fontsize=10)
    ax2.tick_params(colors=COLORS['text'], length=0)
    ax2.set_title('Feature-Importance Weighted Match',
                 color=COLORS['text'], fontsize=14, fontweight='bold', pad=12)
                 
    fig.tight_layout()
    plt.savefig("scratch/combined_test.png")
    print("Saved to scratch/combined_test.png")

if __name__ == "__main__":
    create_combined_plot()
