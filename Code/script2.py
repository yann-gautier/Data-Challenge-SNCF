import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import optuna
from lightgbm import early_stopping, log_evaluation
import gc
import numpy as np

# Chargement et préparation des données

X = pd.read_csv("C:/Users/yanno/OneDrive/Bureau/Data Challenge SNCF/Données/x_train_final.csv")
y = pd.read_csv("C:/Users/yanno/OneDrive/Bureau/Data Challenge SNCF/Données/y_train_final_j5KGWWK.csv")["p0q0"]
X_test = pd.read_csv("C:/Users/yanno/OneDrive/Bureau/Data Challenge SNCF/Données/x_test_final.csv")

# Traitement des colonnes catégorielles
for col in ["train", "gare"]:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# Traitement des dates
X["date"] = pd.to_datetime(X["date"])
X["jour_semaine"] = X["date"].dt.weekday
X["heure"] = X["date"].dt.hour
X["mois"] = X["date"].dt.month

X_test["date"] = pd.to_datetime(X_test["date"])
X_test["jour_semaine"] = X_test["date"].dt.weekday
X_test["heure"] = X_test["date"].dt.hour
X_test["mois"] = X_test["date"].dt.month

# Conversion des dates en timestamp
for col in X.select_dtypes(include=["datetime64[ns]"]):
    X[col] = X[col].astype("int64") // 10**9

for col in X_test.select_dtypes(include=["datetime64[ns]"]):
    X_test[col] = X_test[col].astype("int64") // 10**9

# Nettoyage des noms de colonnes
X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True).str.strip('_')
X_test.columns = X_test.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True).str.strip('_')

# Suppression de colonnes inutiles
if 'Unnamed_0_1' in X.columns:
    X = X.drop('Unnamed_0_1', axis=1)

# Gestion des colonnes catégorielles
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Créer des encoders pour chaque colonne et les sauvegarder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Combiner train et test pour avoir toutes les catégories possibles
    combined_values = pd.concat([X[col].astype(str), X_test[col].astype(str)])
    le.fit(combined_values)
    
    # Transformer train et test avec le même encoder
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    
    label_encoders[col] = le

# Conversion en float32 des colonnes numériques seulement
numeric_cols = X.select_dtypes(include=[np.number]).columns
X[numeric_cols] = X[numeric_cols].astype(np.float32)
X_test[numeric_cols] = X_test[numeric_cols].astype(np.float32)

# Regroupement des classes rares

print("Original unique classes:", len(y.unique()))

# Analyser la distribution des classes
class_counts = pd.Series(y).value_counts().sort_index()
print(f"Class distribution - Min: {class_counts.min()}, Max: {class_counts.max()}")

# Définir un seuil pour les classes rares
MIN_SAMPLES_THRESHOLD = 5  # Ajustable
rare_classes = class_counts[class_counts < MIN_SAMPLES_THRESHOLD].index
normal_classes = class_counts[class_counts >= MIN_SAMPLES_THRESHOLD].index

print(f"Classes with < {MIN_SAMPLES_THRESHOLD} samples: {len(rare_classes)}")
print(f"Classes with >= {MIN_SAMPLES_THRESHOLD} samples: {len(normal_classes)}")

# Stratégie de regroupement: regrouper les classes rares avec leurs voisines les plus proches
def group_rare_classes(y_series, rare_classes, normal_classes, class_counts):
    """
    Regroupe les classes rares avec leurs voisines normales les plus proches
    """
    y_grouped = y_series.copy()
    class_mapping = {}
    
    for rare_class in rare_classes:
        # Trouver la classe normale la plus proche
        distances = np.abs(normal_classes - rare_class)
        closest_normal_class = normal_classes[np.argmin(distances)]
        
        # Remplacer la classe rare par la classe normale la plus proche
        y_grouped = y_grouped.replace(rare_class, closest_normal_class)
        class_mapping[rare_class] = closest_normal_class
        
        print(f"Classe rare {rare_class} ({class_counts[rare_class]} samples) -> Classe {closest_normal_class}")
    
    return y_grouped, class_mapping

# Appliquer le regroupement
if len(rare_classes) > 0:
    y_grouped, class_mapping = group_rare_classes(y, rare_classes, normal_classes, class_counts)
    print(f"\nClasses après regroupement: {len(y_grouped.unique())}")
    print(f"Échantillons regroupés: {len(rare_classes)} classes rares")
else:
    y_grouped = y.copy()
    class_mapping = {}
    print("Aucune classe rare détectée, pas de regroupement nécessaire.")

# Créer un mapping inverse pour les prédictions
inverse_class_mapping = {}
for rare, normal in class_mapping.items():
    if normal not in inverse_class_mapping:
        inverse_class_mapping[normal] = [normal]
    inverse_class_mapping[normal].append(rare)

# Encodage des labels après regroupement
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y_grouped)
print(f"Classes encodées: {len(np.unique(y_encoded))} (de {y_encoded.min()} à {y_encoded.max()})")

# Séparation train/validation - utilisation d'un split aléatoire pour plus de robustesse
X_train, X_ev, y_train, y_ev = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print(f"Train shape: {X_train.shape}, classes: {len(np.unique(y_train))}")
print(f"Validation shape: {X_ev.shape}, classes: {len(np.unique(y_ev))}")

# Optimisation bayésienne Optuna avec MAE

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'mae',
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
        'num_threads': 4,
        'random_state': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    ev_data = lgb.Dataset(X_ev, label=y_ev, reference=train_data)

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

    # Prédictions pour la régression
    y_pred = model.predict(X_ev, num_iteration=model.best_iteration)
    
    # Calculer MAE
    score = mean_absolute_error(y_ev, y_pred)

    # Libération mémoire
    del train_data, ev_data, model, y_pred
    gc.collect()

    return score

# Lancer l'optimisation
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200, show_progress_bar=True)

print("Meilleurs paramètres :", study.best_params)
print("Score (MAE) :", study.best_value)

# Entraînement final avec meilleurs paramètres

best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'metric': 'mae',
    'verbose': -1,
    'random_state': 42
})

train_data_final = lgb.Dataset(X, label=y_encoded)
final_model = lgb.train(
    best_params,
    train_data_final,
    num_boost_round=2000,
    callbacks=[log_evaluation(period=100)]
)

# Prédictions sur le test set avec gestion robuste

y_pred_continuous = final_model.predict(X_test)

# Fonction pour convertir les prédictions continues en classes valides
def continuous_to_valid_class(pred_continuous, le_target):
    """
    Convertit les prédictions continues en classes valides en garantissant
    que toutes les prédictions correspondent à des classes existantes
    """
    # Arrondir aux entiers les plus proches
    pred_rounded = np.round(pred_continuous).astype(int)
    
    # Clipper dans la plage des classes encodées
    min_class = 0
    max_class = len(le_target.classes_) - 1
    pred_clipped = np.clip(pred_rounded, min_class, max_class)
    
    return pred_clipped

# Convertir les prédictions continues en classes
y_pred_encoded = continuous_to_valid_class(y_pred_continuous, le_target)

# Transformer en classes originales (après regroupement)
y_pred_classes = le_target.inverse_transform(y_pred_encoded)

# Optionnel: Post-traitement pour "dé-regrouper" certaines prédictions
# (garder les classes regroupées pour la cohérence)

df_preds = pd.DataFrame({
    "id": X_test.index,
    "prediction": y_pred_classes
})

# Statistiques des prédictions
print(f"\nPrediction statistics:")
print(f"Continuous predictions - Min: {y_pred_continuous.min():.2f}, Max: {y_pred_continuous.max():.2f}")
print(f"Final predictions - Min: {y_pred_classes.min()}, Max: {y_pred_classes.max()}")
print(f"Unique predicted classes: {len(np.unique(y_pred_classes))}")
print(f"All predictions are integers: {all(isinstance(x, (int, np.integer)) for x in y_pred_classes)}")

# Vérifier que toutes les prédictions sont valides
valid_classes = set(y_grouped.unique())
invalid_predictions = set(y_pred_classes) - valid_classes
if invalid_predictions:
    print(f"Warning: {len(invalid_predictions)} invalid predictions found: {invalid_predictions}")
else:
    print("Toutes les prédictions sont valides")

df_preds.to_csv("predictions.csv", index=False)
print("Fichier 'predictions.csv' généré.")