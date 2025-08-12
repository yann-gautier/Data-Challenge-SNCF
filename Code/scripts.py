import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb

# 1. Charger les données
df = pd.read_csv("donnees.csv")

# 2. Séparer features et target
X = df.drop(columns=["retard_minutes"])
y = df["retard_minutes"]

X = pd.read_csv()
y = pd.read_csv()

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Dataset LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# 5. Paramètres LightGBM
params = {
    'objective': 'multiclass',
    'num_class': len(y.unique()),  # nombre de classes
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'verbose': -1
}

# 6. Entraînement
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data],
    num_boost_round=1000,
    early_stopping_rounds=50,
    verbose_eval=100
)

# 7. Prédictions
y_pred_proba = model.predict(X_test)  # probabilités par classe
y_pred = y_pred_proba.argmax(axis=1)  # choisir classe la plus probable

# 8. Évaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
