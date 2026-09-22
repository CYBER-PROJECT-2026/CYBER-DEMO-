import xgboost as xgb
import shap
import numpy as np

X = np.random.randn(100, 24)
y = np.random.randint(0, 4, size=100)

clf = xgb.XGBClassifier(objective="multi:softprob", num_class=4, n_estimators=10, max_depth=3, random_state=42)
clf.fit(X, y)

explainer = shap.TreeExplainer(clf)
shap_values = explainer(X[:10])
print("SHAP values shape:", shap_values.shape)
print("Base values:", explainer.expected_value)
