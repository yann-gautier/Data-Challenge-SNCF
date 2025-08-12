import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import lightgbm as lgb
import optuna
from lightgbm import early_stopping, log_evaluation
import gc
import numpy as np


# -------------------------------
# Chargement et préparation des données
# -------------------------------
X = pd.read_csv("C:/Users/yanno/OneDrive/Bureau/Data Challenge SNCF/Données/x_train_final.csv")
y = pd.read_csv("C:/Users/yanno/OneDrive/Bureau/Data Challenge SNCF/Données/y_train_final_j5KGWWK.csv")["p0q0"]
X_test = pd.read_csv("C:/Users/yanno/OneDrive/Bureau/Data Challenge SNCF/Données/x_test_final.csv")

for col in ["train", "gare"]:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

X["date"] = pd.to_datetime(X["date"])
X["jour_semaine"] = X["date"].dt.weekday
X["heure"] = X["date"].dt.hour
X["mois"] = X["date"].dt.month

X_test["date"] = pd.to_datetime(X_test["date"])
X_test["jour_semaine"] = X_test["date"].dt.weekday
X_test["heure"] = X_test["date"].dt.hour
X_test["mois"] = X_test["date"].dt.month

for col in X.select_dtypes(include=["datetime64[ns]"]):
    X[col] = X[col].astype("int64") // 10**9

for col in X_test.select_dtypes(include=["datetime64[ns]"]):
    X_test[col] = X_test[col].astype("int64") // 10**9

X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True).str.strip('_')
X_test.columns = X_test.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True).str.strip('_')

X = X.drop('Unnamed_0_1', axis=1)

categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Encode categorical variables
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col].astype(str))

categorical_cols2 = X_test.select_dtypes(include=['object', 'category']).columns

le = LabelEncoder()
for col in categorical_cols:
    X_test[col] = le.fit_transform(X_test[col].astype(str))

X = X.astype(np.float32)
X_test = X_test.astype(np.float32)

# Encodage des labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Séparation train/validation
X_train, X_ev, y_train, y_ev = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# -------------------------------
# Optimisation bayésienne Optuna
# -------------------------------
def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': len(y.unique()),
        'metric': 'multi_logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', -1, 20),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'verbose': -1,
        'num_threads': 4  # Limite l'utilisation CPU/RAM
    }

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=["train", "gare"])
    ev_data = lgb.Dataset(X_ev, label=y_ev, categorical_feature=["train", "gare"])

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, ev_data],
        num_boost_round=2000,
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=100)
        ]
    )

    y_pred_proba = model.predict(X_ev)
    score = log_loss(y_ev, y_pred_proba)

    # Libération mémoire
    del train_data, ev_data, model, y_pred_proba
    gc.collect()

    return score

# Lancer l'optimisation
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200, show_progress_bar=True)

print("Meilleurs paramètres :", study.best_params)
print("Score (multi_logloss) :", study.best_value)

# -------------------------------
# Entraînement final avec meilleurs params
# -------------------------------
best_params = study.best_params
best_params.update({
    'objective': 'multiclass',
    'num_class': len(y.unique()),
    'metric': 'multi_logloss',
    'verbose': -1
})

train_data_final = lgb.Dataset(X, label=y_encoded, categorical_feature=["train", "gare"])
final_model = lgb.train(
    best_params,
    train_data_final,
    num_boost_round=2000,
    callbacks=[log_evaluation(period=100)]
)

# -------------------------------
# Prédictions sur le test set
# -------------------------------
y_pred_proba = final_model.predict(X_test)
y_pred = y_pred_proba.argmax(axis=1)
y_pred_classes = le.inverse_transform(y_pred)

df_preds = pd.DataFrame({
    "id": X_test.index,
    "prediction": y_pred_classes
})
df_preds.to_csv("predictions.csv", index=False)
print("Fichier 'predictions.csv' généré.")
