import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb

# 1. Charger les données

X = pd.read_csv("C:/Users/yanno/OneDrive/Bureau/Data Challenge SNCF/Données/x_train_final.csv")
y = pd.read_csv("C:/Users/yanno/OneDrive/Bureau/Data Challenge SNCF/Données/y_train_final_j5KGWWK.csv")["p0q0"]
X_test= pd.read_csv("C:/Users/yanno/OneDrive/Bureau/Data Challenge SNCF/Données/x_test_final.csv")

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

X.columns = (
    X.columns
      .str.replace('[^A-Za-z0-9_]+', '_', regex=True)  # remplace caractères spéciaux par "_"
      .str.strip('_')  # supprime "_" en début/fin
)

X_test.columns = (
    X_test.columns
      .str.replace('[^A-Za-z0-9_]+', '_', regex=True)  # remplace caractères spéciaux par "_"
      .str.strip('_')  # supprime "_" en début/fin
)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Train/test split
X_train, X_ev, y_train, y_ev = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 4. Dataset LightGBM
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=["train", "gare"])
ev_data = lgb.Dataset(X_ev, label=y_ev, categorical_feature=["train", "gare"])

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
from lightgbm import early_stopping, log_evaluation

model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, ev_data],
    num_boost_round=1000,
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=100)  # affichage tous les 100 tours
    ]
)


# 7. Prédictions
y_pred_proba = model.predict(X_test)  # probabilités par classe
y_pred = y_pred_proba.argmax(axis=1)  # choisir classe la plus probable
y_pred_classes = le.inverse_transform(y_pred)
