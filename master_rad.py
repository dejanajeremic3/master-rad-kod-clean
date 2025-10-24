# -*- coding: utf-8 -*-
"""Master rad.ipynb
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, roc_auc_score
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from lightgbm import early_stopping
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
from patsy import dmatrices

# Učitavanje podataka
df = pd.read_excel("Osiguranici_DZO.xlsx")

# Prikaz raspodele podataka
distribution = df['Broj šteta'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
distribution.plot(kind='bar')

plt.xlabel('Broj šteta')
plt.ylabel('Broj osiguranika')
plt.title('Raspodela broja šteta po osiguraniku')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Prikaz broja redova, nedostajućih vrednosti i kolona
print("Broj redova:", df.shape[0])
print("Broj nedostajućih vrednosti:", max(df.isnull().sum()))
print("Kolone:", df.columns)

# Definišemo kolone koje ćemo koristiti
binary_features = [
    'Drugo mišljenje', 'Fizikalna terapija', 'Komplementarna medicina',
    'Oftalmološki pregled i usluge', 'Posebno pokriće u slučaju tumora',
    'Prepisivanje lekova od strane ovlašćenog lekara', 'Sistematski',
    'Stomatološke usluge', 'Vanbolničko i bolničko lečenje',
    'Vanbolničko lečenje',
    'Zdravstvena zaštita trudnica + Troškovi porođaja'
]

numeric_features = [
    'Opšta participacija', 'Bel Medic participacija', 'Medigroup participacija',
    'Van mreže participacija', 'Starost osiguranika'
]

categorical_features = [
    'Stranac', 'Pol osiguranika', 'Veliki grad', 'Interna/Eksterna prodaja'
]

# Formiramo X i y
X = df[binary_features + numeric_features + categorical_features]
y = df["Broj šteta"]

# dummy encoding za kategoričke promenljive
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Uklanjamo redove sa nedostajućim vrednostima
X = X.dropna()
y = y.loc[X.index]

# Podela na train i test skupove
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pripremamo i delimo podatke sa One-hot enkodingom

# Formiramo X i y
X_ohe = df[binary_features + numeric_features + categorical_features]
y_ohe = df["Broj šteta"]

# One-hot enkoding za kategoričke promenljive
X_ohe = pd.get_dummies(X_ohe, columns=categorical_features)

# Uklanjamo redove sa nedostajućim vrednostima vrednostima
X_ohe = X_ohe.dropna()
y_ohe = y_ohe.loc[X_ohe.index]

# Treniranje linearnog modela
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predikcije
y_pred = lr.predict(X_test)

# Evaluacija
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Koeficijenti modela
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

# Crtanje grafika
plt.figure(figsize=(10, 8))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='skyblue')
plt.axvline(x=0, color='black', linestyle='--')
plt.title("Koeficijenti linearne regresije")
plt.xlabel("Vrednost koeficijenta")
plt.tight_layout()
plt.show()

# Reziduali
residuals = y_test - y_pred

# Dijagnostički grafik
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikcija")
plt.ylabel("Rezidual")
plt.title("Dijagnostički grafik: Reziduali u odnosu na predikcije")
plt.tight_layout()

# Dodajemo konstantu
X_sm = sm.add_constant(X)

# Podela na train i test skupove
X_train, X_test, y_train, y_test = train_test_split(X_sm, y, test_size=0.2, random_state=42)

# Konverzija u numeričke vrednosti
X_train = X_train.apply(pd.to_numeric)
y_train = pd.to_numeric(y_train)

X_test = X_test.apply(pd.to_numeric)
X_test = X_test.astype(float)

X_train = X_train.astype(float)

# Kreiranje GLM modela
glm_model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()

# Rezultati modela
print(glm_model.summary())

# Predikcija na osnovu test skupa
y_pred = glm_model.predict(X_test)
y_pred = y_pred.clip(lower=0)

# Evaluacija modela
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Stabilno računanje Puasonove devijacije na test skupu
# Dodajemo malu konstantu da izbegnemo log(0)
epsilon = 1e-10
y_test_safe = np.where(y_test == 0, epsilon, y_test)
y_pred_safe = np.where(y_pred == 0, epsilon, y_pred)

deviance = 2 * np.sum(y_test_safe * np.log(y_test_safe / y_pred_safe) - (y_test_safe - y_pred_safe))
mean_deviance = deviance / len(y_test)

print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'Srednja devijacija: {mean_deviance}')

# Grafik koeficijenata
coefficients = glm_model.params[1:]  # Ignoriši konstantu
coefficients = coefficients.sort_values(ascending=False)

# Filtriranje koeficijente da isključimo 'Pol osiguranika_grupni', zbog malo instanci u ovoj grupi
filtered_coefficients = coefficients[coefficients.index != 'Pol osiguranika_grupni']

plt.figure(figsize=(10, 8))
sns.barplot(x=filtered_coefficients.values, y=filtered_coefficients.index, palette='Blues_d')
plt.title('Koeficijenti GLM modela')
plt.xlabel('Vrednost koeficijenta')
plt.show()

# Reziduali
residuals = y_test - y_pred

# Dijagnostički grafik
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikcija")
plt.ylabel("Rezidual")
plt.title("Dijagnostički grafik: Reziduali u odnosu na predikcije (Poason)")
plt.tight_layout()

# 1. Podela na trening i test skup - bez konstante
X_train, X_test, y_train, y_test = train_test_split(X_ohe, y_ohe, test_size=0.2, random_state=42)

# 2. Konverzija u numeričke vrednosti
X_train = X_train.apply(pd.to_numeric)
y_train = pd.to_numeric(y_train)

X_test = X_test.apply(pd.to_numeric)
X_test = X_test.astype(float)

X_train = X_train.astype(float)

# 3. Treniranje modela Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# 4. Predikcija
y_pred = rf_model.predict(X_test)

# 5. Evaluacija modela
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R^2: {r2}")

# Uzimamo važnost prediktora
importances = pd.Series(rf_model.feature_importances_, index=X_ohe.columns)
importances_sorted = importances.sort_values(ascending=False)[:20]  # top 20 najvažnijih

# Crtanje grafika
plt.figure(figsize=(10, 6))
importances_sorted.plot(kind='barh')
plt.gca().invert_yaxis()
plt.xlabel('Važnost prediktora')
plt.title('20 najvažnijih nezavisnih promenljivih (Random Forest)')
plt.show()

# Lista naziva kolona koje želimo da prikažemo
features_to_plot = [
    'Starost osiguranika',
    'Pol osiguranika_F',
    'Veliki grad_Beograd'
]

# Crtanje PDP i ICE
PartialDependenceDisplay.from_estimator(
    rf_model,
    X_train,
    features_to_plot,
    kind='both',       # PDP + ICE
    grid_resolution=50,
    subsample=100,     # koristimo 100 slučajnih osiguranika za ICE (da bi grafik bio pregledniji)
    random_state=42
)
plt.suptitle('Partial Dependence i ICE grafici (Random Forest)', fontsize=14)
plt.tight_layout()
plt.show()

def plot_ale_1d(model, X, feature, bins=20):
    """
    Ručna implementacija ALE za neprekidnu promenljivu
    """
    X = X.copy()
    values = X[feature]
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(values, quantiles)
    bin_edges = np.unique(bin_edges)

    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    effects = []

    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        in_bin = (values >= lower) & (values < upper)
        if in_bin.sum() == 0:
            effects.append(0)
            continue

        X_lower = X.copy()
        X_upper = X.copy()
        X_lower.loc[in_bin, feature] = lower
        X_upper.loc[in_bin, feature] = upper

        pred_diff = model.predict(X_upper) - model.predict(X_lower)
        effects.append(np.mean(pred_diff))

    # Akumulacija (integracija) i centriranje
    accumulated = np.cumsum(effects)
    accumulated -= np.mean(accumulated)

    # Crtanje grafika
    plt.figure(figsize=(6, 4))
    plt.plot(centers, accumulated, marker='o')
    plt.xlabel(feature)
    plt.ylabel("ALE efekt")
    plt.title(f"ALE plot – {feature}")
    plt.grid(True)
    plt.show()

def plot_ale_binary(model, X, feature):
    """
    ALE za binarnu promenljivu: dva stanja (0 i 1)
    """
    X = X.copy()
    values = X[feature]

    if not set(values.unique()).issubset({0, 1}):
        raise ValueError("Očekuje se binarna promenljiva (samo 0 i 1)")

    effects = []

    # Samo osiguranici kod kojih je vrednost promenljive = 0
    in_bin = (values == 0)
    if in_bin.sum() == 0:
        effects.append(0)
    else:
        X0 = X.copy()
        X1 = X.copy()
        X0.loc[in_bin, feature] = 0
        X1.loc[in_bin, feature] = 1
        pred_diff = model.predict(X1) - model.predict(X0)
        effects.append(np.mean(pred_diff))

    # Efekat za prelazak sa 0 → 1, centriramo oko 0
    accumulated = np.array([0, effects[0]])
    accumulated -= np.mean(accumulated)

    # Crtanje grafika
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], accumulated, marker='o')
    plt.xticks([0, 1])
    plt.xlabel(feature)
    plt.ylabel("ALE efekt")
    plt.title(f"ALE plot – {feature}")
    plt.grid(True)
    plt.show()

# Crtanje ALE grafika za odabrane promenljive
plot_ale_1d(rf_model, X_train, feature='Starost osiguranika', bins=20)
plot_ale_binary(rf_model, X_train, 'Pol osiguranika_F')
plot_ale_binary(rf_model, X_train, 'Veliki grad_Beograd')

# Inicijalizacija SHAP explainer-a za modele zasnovane na stablima odlučivanja
explainer = shap.Explainer(rf_model, X_train)

# Izračunavanje Šeplijevih vrednosti za uzorak (npr. prvih 100 primera radi brzine)
shap_values = explainer(X_train.iloc[:100])

# Grafik raspodele uticaja (globalna važnost i efekat)
shap.plots.beeswarm(shap_values)

# Bar plot sa srednjim Šeplijevim vrednostima
shap.plots.bar(shap_values)

# Lokalna interpretacija jedne instance (npr. prvi osiguranik)
shap.plots.waterfall(shap_values[0])

# 1. Podela na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(X_ohe, y_ohe, test_size=0.2, random_state=42)

# 2. Treniranje neuronske mreže
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)

# 3. Predikcija
y_pred = mlp_model.predict(X_test)

# 4. Evaluacija
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

# Računanje permutation importance
perm_importance = permutation_importance(mlp_model, X_test, y_test, n_repeats=10, random_state=42)

# Pretvaranje u DataFrame za lakši prikaz
importance_df = pd.DataFrame({
    'Feature': X_ohe.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

# Prikaz prvih 20
top_features = importance_df.head(30)

# Crtanje grafika
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'])
plt.gca().invert_yaxis()
plt.xlabel('Permutation importance')
plt.title('Feature importance – neuronska mreža')
plt.tight_layout()
plt.show()

# Lista promenljivih za koje prikazujemo PDP + ICE
features_to_plot = [
    'Medigroup participacija',
    'Bel Medic participacija',
    'Posebno pokriće u slučaju tumora',
    'Starost osiguranika'
]

# 2x2 raspored
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# Crtanje PDP + ICE
PartialDependenceDisplay.from_estimator(
    mlp_model,
    X_train,
    features_to_plot,
    kind='both',
    grid_resolution=50,
    subsample=100,
    ax=ax
)

plt.suptitle("Partial Dependence i ICE prikaz (neuronska mreža)", fontsize=14)
plt.tight_layout()
plt.show()

# ALE za neprekidne promenljive

plot_ale_1d(mlp_model, X_train, 'Medigroup participacija')
plot_ale_1d(mlp_model, X_train, 'Bel Medic participacija')
plot_ale_1d(mlp_model, X_train, 'Starost osiguranika')

# ALE za binarne promenljive
plot_ale_binary(mlp_model, X_train, 'Posebno pokriće u slučaju tumora')

# Uzimamo manji uzorak radi brzine
X_sample = X_train.sample(200, random_state=42)

# Inicijalizacija KernelExplainer-a
explainer = shap.KernelExplainer(mlp_model.predict, X_sample)

# Šeplijeve vrednosti za uzorak test skupa
X_eval = X_test.sample(200, random_state=42)
shap_values = explainer.shap_values(X_eval)

# Grafik raspodele uticaja Šeplijevih vrednosti
shap.summary_plot(shap_values, X_eval, max_display=10)

# Bar plot sa srednjim Šeplijevim vrednostima
shap.summary_plot(shap_values, X_eval, plot_type="bar", max_display=10)

# Lokalna interpretacija jedne instance (npr. prvi osiguranik)
shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value, shap_values[0], X_eval.iloc[0]
)

# Kreiranje LightGBM modela
lgb_model = lgb.LGBMRegressor(random_state=42)

# Treniranje modela
lgb_model.fit(X_train, y_train)

# Predikcija
y_pred = lgb_model.predict(X_test)

# Evaluacija
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R²: {r2}")

# Permutation importance
perm = permutation_importance(lgb_model, X_test, y_test, n_repeats=10, random_state=42)

# Tabela rezultata
importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': perm.importances_mean
}).sort_values(by='importance', ascending=False)

# Prikaz rezultata
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.gca().invert_yaxis()
plt.title('Permutation Importance (LightGBM)')
plt.xlabel('Mean Importance')
plt.tight_layout()
plt.show()

# Definišemo prostor za pretragu hiperparametara
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [15, 31, 50, 70],
    'max_depth': [3, 5, 7, -1],
    'min_child_samples': [5, 10, 20],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_grid,
    n_iter=30,  # broj kombinacija koje ćemo isprobati
    scoring='r2',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Treniranje modela
random_search.fit(X_train, y_train)

# Najbolji rezultati
print("Najbolji parametri:", random_search.best_params_)
print("Najbolji MAE (CV):", -random_search.best_score_)

# Model sa najboljim parametrima
best_lgb = random_search.best_estimator_

# Predikcije
y_pred = best_lgb.predict(X_test)

# Evaluacija
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R²: {r2}")

# targetiramo promenljive na osnovu svih prethodnih rezultata interpretacije

# Kopiramo bazu da ne pregazimo original
X_train_fe = X_train.copy()
X_test_fe = X_test.copy()

# Interakcije i transformacije

# Interakcija: starost * pol
X_train_fe['starost_x_pol'] = X_train_fe['Starost osiguranika'] * X_train_fe['Pol osiguranika_F']
X_test_fe['starost_x_pol'] = X_test_fe['Starost osiguranika'] * X_test_fe['Pol osiguranika_F']

# Suma participacija
X_train_fe['total_participacija'] = X_train_fe['Bel Medic participacija'] + X_train_fe['Medigroup participacija']
X_test_fe['total_participacija'] = X_test_fe['Bel Medic participacija'] + X_test_fe['Medigroup participacija']

# Participacija * vanbolničko i bolničko lečenje
X_train_fe['belmedic_x_vanbol'] = X_train_fe['Bel Medic participacija'] * X_train_fe['Vanbolničko i bolničko lečenje']
X_test_fe['belmedic_x_vanbol'] = X_test_fe['Bel Medic participacija'] * X_test_fe['Vanbolničko i bolničko lečenje']

# Interna prodaja * grad grad_Beograd
if 'Veliki grad_Beograd' in X_train_fe.columns:
    X_train_fe['interna_x_beograd'] = X_train_fe['Interna/Eksterna prodaja_Interna'] * X_train_fe['Veliki grad_Beograd']
    X_test_fe['interna_x_beograd'] = X_test_fe['Interna/Eksterna prodaja_Interna'] * X_test_fe['Veliki grad_Beograd']

# Log transformacija starosti
import numpy as np
X_train_fe['log_starost'] = np.log1p(X_train_fe['Starost osiguranika'])
X_test_fe['log_starost'] = np.log1p(X_test_fe['Starost osiguranika'])

best_model = lgb.LGBMRegressor(**random_search.best_params_)  # koristimo tunirane parametre
best_model.fit(X_train_fe, y_train)
y_pred = best_model.predict(X_test_fe)

# Evaluacija
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Dvostepeni pristup

# 1. Binarna klasifikacija: da li je osiguranik imao štetu
y_bin = (y > 0).astype(int)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

# Podela na trening i test skupove
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.2, random_state=42)

# Logistička regresija
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_bin, y_train_bin)
y_proba = log_model.predict_proba(X_test_bin)[:, 1]

# AUC i ROC
from matplotlib import pyplot as plt
fpr, tpr, _ = roc_curve(y_test_bin, y_proba)
auc = roc_auc_score(y_test_bin, y_proba)

# Grafik
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC kriva - GLM klasifikacija')
plt.legend()
plt.grid(True)
plt.show()

# Priprema ciljne promenljive i podela na trening i test skupove
y_bin2 = (y_ohe > 0).astype(int)
X_train_bin2, X_test_bin2, y_train_bin2, y_test_bin2 = train_test_split(X_ohe, y_bin2, test_size=0.2, random_state=42)

# Logistička regresija – original
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_bin, y_train_bin)
y_pred_bin = log_model.predict(X_test_bin)
y_proba_bin = log_model.predict_proba(X_test_bin)[:, 1]

# Logistička regresija sa balansiranjem klasa – class_weight='balanced'
log_bal = LogisticRegression(max_iter=1000, class_weight='balanced')
log_bal.fit(X_train_bin, y_train_bin)
y_pred_bal = log_bal.predict(X_test_bin)
y_proba_bal = log_bal.predict_proba(X_test_bin)[:, 1]

# SMOTE + Random Forest
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_bin2, y_train_bin2)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_sm, y_train_sm)
y_pred_rf = rf_model.predict(X_test_bin2)
y_proba_rf = rf_model.predict_proba(X_test_bin2)[:, 1]

# Evaluacija
def eval_class(y_true, y_pred, y_proba):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1 Score": round(f1_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_proba), 4),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist()
    }

results = {
    "GLM original": eval_class(y_test_bin, y_pred_bin, y_proba_bin),
    "GLM balanced": eval_class(y_test_bin, y_pred_bal, y_proba_bal),
    "RF + SMOTE": eval_class(y_test_bin, y_pred_rf, y_proba_rf)
}

# ROC krive
fpr_o, tpr_o, _ = roc_curve(y_test_bin, y_proba_bin)
fpr_b, tpr_b, _ = roc_curve(y_test_bin, y_proba_bal)
fpr_rf, tpr_rf, _ = roc_curve(y_test_bin2, y_proba_rf)

# Prikaz rezultata
plt.figure(figsize=(8,6))
plt.plot(fpr_o, tpr_o, label=f"GLM original (AUC={results['GLM original']['AUC']})")
plt.plot(fpr_b, tpr_b, label=f"GLM balanced (AUC={results['GLM balanced']['AUC']})")
plt.plot(fpr_rf, tpr_rf, label=f"RF + SMOTE (AUC={results['RF + SMOTE']['AUC']})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC kriva – Klasifikacija šteta (Hurdle - korak 1)")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Ispis metrika
import pandas as pd
pd.DataFrame(results).T

# Treniranje LGBM klasifikatora
lgb_clf = LGBMClassifier(random_state=42)
lgb_clf.fit(X_train_bin2, y_train_bin2)

# Predikcije
y_pred_lgb_class = lgb_clf.predict(X_test_bin2)
y_proba_lgb_class = lgb_clf.predict_proba(X_test_bin2)[:, 1]

# Evaluacija
acc_lgb = accuracy_score(y_test_bin2, y_pred_lgb_class)
prec_lgb = precision_score(y_test_bin2, y_pred_lgb_class)
rec_lgb = recall_score(y_test_bin2, y_pred_lgb_class)
f1_lgb = f1_score(y_test_bin2, y_pred_lgb_class)
auc_lgb = roc_auc_score(y_test_bin2, y_proba_lgb_class)
conf_lgb = confusion_matrix(y_test_bin2, y_pred_lgb_class)

# ROC kriva
fpr_lgb, tpr_lgb, _ = roc_curve(y_test_bin2, y_proba_lgb_class)

# Prikaz
plt.figure(figsize=(8, 6))
plt.plot(fpr_lgb, tpr_lgb, label=f"LightGBM (AUC = {auc_lgb:.4f})", color='darkorange')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC kriva – LightGBM klasifikacija šteta")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Ispis metrika
{
    "Accuracy": round(acc_lgb, 4),
    "Precision": round(prec_lgb, 4),
    "Recall": round(rec_lgb, 4),
    "F1 Score": round(f1_lgb, 4),
    "AUC": round(auc_lgb, 4),
    "Confusion Matrix": conf_lgb.tolist()
}

# Definišemo prostor za pretragu hiperparametara
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [15, 31, 50],
    'max_depth': [-1, 5, 10],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# RandomizedSearchCV
lgb_model = LGBMClassifier(random_state=42)
search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_grid,
    scoring='roc_auc',
    cv=5,
    n_iter=25,
    verbose=1,
    n_jobs=-1
)

# Treniranje modela
search.fit(X_train_bin2, y_train_bin2)

# Najbolji rezultati
print("Najbolji parametri:", search.best_params_)
print("Najbolji AUC (CV):", search.best_score_)

# Model sa najboljim parametrima
best_lgb = search.best_estimator_

# Predikcije
y_pred_best = best_lgb.predict(X_test_bin2)
y_proba_best = best_lgb.predict_proba(X_test_bin2)[:, 1]

# Metrike
acc_best = accuracy_score(y_test_bin2, y_pred_best)
prec_best = precision_score(y_test_bin2, y_pred_best)
rec_best = recall_score(y_test_bin2, y_pred_best)
f1_best = f1_score(y_test_bin2, y_pred_best)
auc_best = roc_auc_score(y_test_bin2, y_proba_best)
conf_best = confusion_matrix(y_test_bin2, y_pred_best)

# ROC kriva
fpr_best, tpr_best, _ = roc_curve(y_test_bin2, y_proba_best)

# Prikaz
plt.figure(figsize=(8, 6))
plt.plot(fpr_best, tpr_best, label=f"Tunirani LGBM (AUC = {auc_best:.4f})", color='darkblue')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC kriva – Optimizovan LightGBM")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Ispis rezultata
{
    "Accuracy": round(acc_best, 4),
    "Precision": round(prec_best, 4),
    "Recall": round(rec_best, 4),
    "F1 Score": round(f1_best, 4),
    "AUC": round(auc_best, 4),
    "Confusion Matrix": conf_best.tolist()
}

# Optimizacija praga za F1 skor

thresholds = np.arange(0.1, 0.91, 0.01)
best_thresh = 0.5
best_f1 = 0

for t in thresholds:
    preds = (y_proba_best >= t).astype(int)
    score = f1_score(y_test_bin2, preds)
    if score > best_f1:
        best_f1 = score
        best_thresh = t

print(f"Najbolji prag: {best_thresh:.2f}")
print(f"Najviši F1-score: {best_f1:.4f}")

from sklearn.metrics import roc_curve, auc

# Trenirani model koji ti je vratio RandomizedSearchCV
final_model = search.best_estimator_

# Koristimo optimalni prag za predikciju
optimal_threshold = 0.31

# Predikcije sa optimizovanim pragom
y_proba_final = final_model.predict_proba(X_test_bin2)[:, 1]
y_pred_final = (y_proba_final >= 0.31).astype(int)

# Metrike
acc_final = accuracy_score(y_test_bin2, y_pred_final)
prec_final = precision_score(y_test_bin2, y_pred_final)
rec_final = recall_score(y_test_bin2, y_pred_final)
f1_final = f1_score(y_test_bin2, y_pred_final)
auc_final = roc_auc_score(y_test_bin2, y_proba_final)
conf_final = confusion_matrix(y_test_bin2, y_pred_final)

# ROC kriva
fpr, tpr, _ = roc_curve(y_test_bin2, y_proba_final)
roc_auc = auc(fpr, tpr)

# Grafik
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"LightGBM (AUC = {roc_auc:.4f})", color='navy')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC kriva – Konačni LightGBM klasifikator (prvi korak Hurdle modela)")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Ispis
print("=== Konačne metrike klasifikacije (LightGBM + prag 0.34) ===")
print(f"Accuracy:  {acc_final:.4f}")
print(f"Precision: {prec_final:.4f}")
print(f"Recall:    {rec_final:.4f}")
print(f"F1 Score:  {f1_final:.4f}")
print(f"AUC:       {auc_final:.4f}")
print(f"Confusion Matrix:\n{conf_final}")

# Precision-Recall (PR) kriva kao dodatak ROC-u, posebno korisna u slučaju neuravnoteženih klasa

# Verovatnoće predikcije
y_proba_final = final_model.predict_proba(X_test_bin2)[:, 1]

# Precision-Recall vrednosti
precision, recall, _ = precision_recall_curve(y_test_bin2, y_proba_final)
avg_precision = average_precision_score(y_test_bin2, y_proba_final)

# Grafik
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2,
         label=f'PR kriva (AP = {avg_precision:.4f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall kriva – Tunirani LightGBM")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# interpretacija finalnog modela za prvi deo dvostepenog pristupa
# Uzimamo važnost prediktora

# Kreiramo DataFrame sa važnostima
feature_importance = pd.DataFrame({
    'Feature': final_model.feature_name_,
    'Importance': final_model.feature_importances_
})

# Sortiramo po važnosti
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Grafik
plt.figure(figsize=(10, 6))
importances_sorted.plot(kind='barh')
plt.gca().invert_yaxis()
plt.xlabel('Važnost prediktora')
plt.title('20 najvažnijih nezavisnih promenljivih (Konačan LightGBM model)')
plt.show()

def plot_ale_binary(model, X, feature_name, bins=20, class_index=1):
    """
    Ručno crtanje ALE za binarnu klasifikaciju, za numeričku ili binarnu promenljivu.
    """
    x = X.copy()
    feat = x[feature_name]

    if feat.nunique() == 2:  # Binarna promenljiva
        values = sorted(feat.unique())
        effects = []
        for val in values:
            x_temp = x.copy()
            x_temp[feature_name] = val
            preds = model.predict_proba(x_temp)[:, class_index]
            effects.append(np.mean(preds))
        centered_ale = np.array(effects) - np.mean(effects)

        plt.plot(values, centered_ale, marker='o')
        plt.xticks(values)
        plt.xlabel(feature_name)
        plt.ylabel("ALE")
        plt.title(f"ALE za '{feature_name}' (binarna)")
        plt.grid(True)
        plt.show()

    else:  # Numerička promenljiva
        quantiles = np.linspace(0, 1, bins + 1)
        bin_edges = np.quantile(feat, quantiles)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        effects = []
        for i in range(len(bin_edges) - 1):
            x_lower = x.copy()
            x_upper = x.copy()

            mask = (feat >= bin_edges[i]) & (feat < bin_edges[i+1])
            if mask.sum() == 0:
                effects.append(0)
                continue

            x_lower.loc[mask, feature_name] = bin_edges[i]
            x_upper.loc[mask, feature_name] = bin_edges[i+1]

            preds_upper = model.predict_proba(x_upper)[:, class_index]
            preds_lower = model.predict_proba(x_lower)[:, class_index]

            effects.append(np.mean(preds_upper[mask] - preds_lower[mask]))

        ale_values = np.cumsum(effects)
        centered_ale = ale_values - np.mean(ale_values)

        plt.plot(bin_centers, centered_ale, marker='o')
        plt.xlabel(feature_name)
        plt.ylabel("ALE")
        plt.title(f"ALE za '{feature_name}'")
        plt.grid(True)
        plt.show()

plot_ale_binary(best_lgb, X_train_bin2, 'Pol osiguranika_F')
plot_ale_binary(best_lgb, X_train_bin2, 'Starost osiguranika')
plot_ale_binary(best_lgb, X_train_bin2, 'Veliki grad_Beograd')
plot_ale_binary(best_lgb, X_train_bin2, 'Bel Medic participacija')

# Uzimamo manji uzorak radi brzine
X_sample = X_train_bin2.sample(200, random_state=42)

# Inicijalizacija SHAP Explainer-a za LightGBM
explainer = shap.TreeExplainer(best_lgb)

# Šeplijeve vrednosti za uzorak
shap_values = explainer.shap_values(X_sample)

# Grafik raspodele uticaja Šeplijevih vrednosti
shap.summary_plot(shap_values, X_sample, max_display=10)

# Bar plot sa srednjim Šeplijevim vrednostima
shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=10)

# Lokalna interpretacija jedne instance (npr. deseti osiguranik)
shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value, shap_values[10], X_sample.iloc[10]
)

# Sada prelazimo na drugi deo dvostepenog pristupa, modelovanje očekivanog broja šteta za osiguranike koji su prepoznati kao rizični pomoću prvog modela
# prvo pravimo GLM model
import statsmodels.api as sm

# Filtriranje osiguranika koji su imali bar jednu štetu
df_pos = df[df['Broj šteta'] > 0].copy()

# Priprema promenljivih
# Kategoričke promenljive će biti automatski obrađene preko patsy (npr. C(Pol osiguranika))
formula = """Q('Broj šteta') ~
             Q('Starost osiguranika') +
             C(Q('Pol osiguranika')) +
             C(Stranac) +
             Q('Bel Medic participacija') +
             Q('Medigroup participacija') +
             Q('Veliki grad') +
             Q('Fizikalna terapija') +
             Q('Vanbolničko i bolničko lečenje') +
             Q('Interna/Eksterna prodaja')
          """

# Priprema matrica dizajna
y, X = dmatrices(formula, data=df_pos, return_type='dataframe')

# Treniranje negativnog binomnog modela
nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()

# Rezultati
print(nb_model.summary())

# Predikcije i uporedni prikaz
df_pos['Predikcija'] = nb_model.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(df_pos['Broj šteta'], df_pos['Predikcija'], alpha=0.5)
plt.xlabel("Stvarni broj šteta")
plt.ylabel("Predikcija modela")
plt.title("NB model: Stvarni vs. predikovani broj šteta")
plt.grid(True)
plt.tight_layout()
plt.show()

# Uklanjanje redova sa NaN vrednostima u y_true ili y_pred
eval_df = df_pos[['Broj šteta', 'Predikcija']].dropna()

# Prave i predikovane vrednosti nakon čišćenja
y_true = eval_df['Broj šteta']
y_pred = eval_df['Predikcija']

# Evaluacione metrike
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f"MAE (Srednja apsolutna greška): {mae:.4f}")
print(f"RMSE (Korenski srednji kvadratni error): {rmse:.4f}")
print(f"R² score: {r2:.4f}")

# Histogram reziduala
residuals = y_true - y_pred

plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.title("Distribucija reziduala (stvarno - predikcija)")
plt.xlabel("Rezidual")
plt.ylabel("Frekvencija")
plt.grid(True)
plt.tight_layout()
plt.show()

# sada pokušavamo LightGMB regresiju

# Filtriranje samo osiguranika sa bar jednom štetom
df_pos = df[df['Broj šteta'] > 0].copy()

# Selekcija prediktora i targeta
# Prvo enkodujemo kategorijske promenljive
df_pos['Pol osiguranika'] = df_pos['Pol osiguranika'].astype('category')
df_pos['Stranac'] = df_pos['Stranac'].astype('category')
df_pos['Interna/Eksterna prodaja'] = df_pos['Interna/Eksterna prodaja'].astype('category')
df_pos['Veliki grad'] = df_pos['Veliki grad'].astype('category')

# Lista prediktora
features = [
    'Starost osiguranika',
    'Pol osiguranika',
    'Stranac',
    'Interna/Eksterna prodaja',
    'Veliki grad',
    'Fizikalna terapija',
    'Sistematski',
    'Drugo mišljenje',
    'Stomatološke usluge',
    'Komplementarna medicina',
    'Oftalmološki pregled i usluge',
    'Posebno pokriće u slučaju tumora',
    'Prepisivanje lekova od strane ovlašćenog lekara',
    'Vanbolničko lečenje',
    'Vanbolničko i bolničko lečenje',
    'Opšta participacija',
    'Bel Medic participacija',
    'Medigroup participacija'
]

X = df_pos[features]
y = df_pos['Broj šteta']

# Podela na trening i test skupove
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treniranje LightGBM Regressora
lgb_model = LGBMRegressor(random_state=42)
lgb_model.fit(X_train, y_train)

# Predikcija
y_pred = lgb_model.predict(X_test)

# Evaluacija
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² score: {r2:.4f}")

# Grafik
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Stvarni broj šteta")
plt.ylabel("Predikcija modela")
plt.title("LightGBM regresor: Stvarni vs. predikovani broj šteta")
plt.grid(True)
plt.tight_layout()
plt.show()

# sada ćemo pokušati sa logaritamskom transformacijom ciljne promenljive

# Učitavanje i filtriranje podataka
df_pos = df[df['Broj šteta'] > 0].copy()

# Log-transformacija ciljne promenljive
df_pos['Broj šteta log'] = np.log1p(df_pos['Broj šteta'])  # log(1 + x)

# Selekcija prediktora i targeta
# Prvo enkodujemo kategorijske promenljive
df_pos['Pol osiguranika'] = df_pos['Pol osiguranika'].astype('category')
df_pos['Stranac'] = df_pos['Stranac'].astype('category')
df_pos['Interna/Eksterna prodaja'] = df_pos['Interna/Eksterna prodaja'].astype('category')
df_pos['Veliki grad'] = df_pos['Veliki grad'].astype('category')

# Lista nezavisnih promenljivih
features = [
    'Starost osiguranika',
    'Pol osiguranika',
    'Stranac',
    'Interna/Eksterna prodaja',
    'Veliki grad',
    'Fizikalna terapija',
    'Sistematski',
    'Drugo mišljenje',
    'Stomatološke usluge',
    'Komplementarna medicina',
    'Oftalmološki pregled i usluge',
    'Posebno pokriće u slučaju tumora',
    'Prepisivanje lekova od strane ovlašćenog lekara',
    'Vanbolničko lečenje',
    'Vanbolničko i bolničko lečenje',
    'Opšta participacija',
    'Bel Medic participacija',
    'Medigroup participacija'
]

X = df_pos[features]
y_log = df_pos['Broj šteta log']

# Podela na trening i test skupove
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Treniranje LightGBM modela
model_log = LGBMRegressor(random_state=42)
model_log.fit(X_train, y_train_log)

# Predikcija u log-skali
y_pred_log = model_log.predict(X_test)

# Vraćanje u originalnu skalu
y_test = np.expm1(y_test_log)       # exp(y) - 1
y_pred = np.expm1(y_pred_log)

# Evaluacija
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R² score: {r2}")

# Grafik stvarni podaci vs. predikcija
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Stvarni broj šteta")
plt.ylabel("Predikcija (LightGBM, log-transformisano)")
plt.title("Log-transformisana regresija: Stvarni vs. predikovani broj šteta")
plt.grid(True)
plt.tight_layout()
plt.show()

# Sada prelazimo na klasifikaciju, regresija ne odgovara problemu
# Multi-class klasifikacija broja šteta sa 3 klase

# Učitavanje i filtriranje samo osiguranika sa bar jednom štetom
df_pos = df[df['Broj šteta'] > 0].copy()

# Grupisanje u klase: 1–3 / 4–6 / 7+
def map_class(x):
    if x in [1, 2, 3]:
        return 0  # umeren rizik
    elif x in [4, 5, 6]:
        return 1  # veći rizik
    else:
        return 2  # najveći rizik

df_pos['Claim_Group'] = df_pos['Broj šteta'].apply(map_class)

#  Pretvaranje kategoričkih promenljivih
df_pos['Pol osiguranika'] = df_pos['Pol osiguranika'].astype('category')
df_pos['Stranac'] = df_pos['Stranac'].astype('category')
df_pos['Interna/Eksterna prodaja'] = df_pos['Interna/Eksterna prodaja'].astype('category')
df_pos['Veliki grad'] = df_pos['Veliki grad'].astype('category')

# Nezavisne promenljive
features = [
    'Starost osiguranika',
    'Pol osiguranika',
    'Stranac',
    'Interna/Eksterna prodaja',
    'Fizikalna terapija',
    'Sistematski',
    'Drugo mišljenje',
    'Stomatološke usluge',
    'Komplementarna medicina',
    'Oftalmološki pregled i usluge',
    'Posebno pokriće u slučaju tumora',
    'Prepisivanje lekova od strane ovlašćenog lekara',
    'Vanbolničko lečenje',
    'Vanbolničko i bolničko lečenje',
    'Opšta participacija',
    'Bel Medic participacija',
    'Medigroup participacija',
    'Veliki grad'
]

X = df_pos[features]
y = df_pos['Claim_Group']

# One-hot encoding svih kategoričkih promenljivih
X = pd.get_dummies(X, columns=[
    'Pol osiguranika',
    'Stranac',
    'Interna/Eksterna prodaja',
    'Veliki grad'
], drop_first=False)

# Podela na trening i test skupove
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Treniranje LightGBM klasifikatora
model = LGBMClassifier(objective='multiclass', random_state=42)
model.fit(X_train, y_train)

# Predikcija
y_pred = model.predict(X_test)

# Evaluacija
print(classification_report(y_test, y_pred, zero_division=0))

# Konfuziona matrica
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel('Predikcija')
plt.ylabel('Stvarno')
plt.title('Confusion Matrix – Grupisana klasifikacija broja šteta (3 klase)')
plt.tight_layout()
plt.show()

# Sada multi-class klasifikacija sa 2 klase

# Učitavanje i filtriranje
df_pos = df[df['Broj šteta'] > 0].copy()

# Grupisanje u dve klase: 0 = 1–3 štete, 1 = 4+
df_pos['Claim_Binary'] = df_pos['Broj šteta'].apply(lambda x: 0 if x <= 3 else 1)

# Lista nezavisnih promenljivih
features = [
    'Starost osiguranika',
    'Pol osiguranika',
    'Stranac',
    'Interna/Eksterna prodaja',
    'Fizikalna terapija',
    'Sistematski',
    'Drugo mišljenje',
    'Stomatološke usluge',
    'Komplementarna medicina',
    'Oftalmološki pregled i usluge',
    'Posebno pokriće u slučaju tumora',
    'Prepisivanje lekova od strane ovlašćenog lekara',
    'Vanbolničko lečenje',
    'Vanbolničko i bolničko lečenje',
    'Opšta participacija',
    'Bel Medic participacija',
    'Medigroup participacija',
    'Veliki grad'
]

# Selektovanje X i y
X = df_pos[features]
y = df_pos['Claim_Binary']

# One-hot encoding svih kategoričkih promenljivih
X = pd.get_dummies(X, columns=[
    'Pol osiguranika',
    'Stranac',
    'Interna/Eksterna prodaja',
    'Veliki grad'
], drop_first=False)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Treniranje modela
model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# Predikcija
y_pred = model.predict(X_test)

# Evaluacija
print(classification_report(y_test, y_pred, zero_division=0))

# Matrica konfuzije
cm = confusion_matrix(y_test, y_pred)
labels = ['1–3 (umereni)', '4+ (veći)']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predikcija")
plt.ylabel("Stvarno")
plt.title("Confusion Matrix – Binarna klasifikacija broja šteta (2 klase)")
plt.tight_layout()
plt.show()

# Inicijalizacija SHAP Explainer-a
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train.iloc[:100])

# Grafik raspodele uticaja Šeplijevih vrednosti
shap.summary_plot(shap_values, X_train.iloc[:100], max_display=10)

# Bar plot sa srednjim Šeplijevim vrednostima
shap.summary_plot(shap_values, X_train.iloc[:100], plot_type="bar", max_display=10)

# Lokalna interpretacija jedne instance
shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value,
    shap_values[0],
    feature_names=X_train.columns
)

# Konačna predikcija
# Pravimo kopiju originalnog df (sačuvan df_copy na početku)
df_all_1 = df_copy.copy()

# One-hot encoding za step 1 – kao u trening skupu
df_all_1 = pd.get_dummies(df_all_1, columns=[
    'Pol osiguranika',
    'Stranac',
    'Interna/Eksterna prodaja',
    'Veliki grad'
], drop_first=True)
df_all_1.columns = df_all_1.columns.str.replace(' ', '_')

# Kolone korišćene u step 1
X_all_step1 = df_all_1[final_model.feature_name_]

# Napravi kopiju za step 2
df_all_2 = df_copy.copy()

# Pretvaranje u kategoričke
df_all_2['Pol osiguranika'] = df_all_2['Pol osiguranika'].astype('category')
df_all_2['Stranac'] = df_all_2['Stranac'].astype('category')
df_all_2['Interna/Eksterna prodaja'] = df_all_2['Interna/Eksterna prodaja'].astype('category')
df_all_2['Veliki grad'] = df_all_2['Veliki grad'].astype('category')
df_all_2.columns = df_all_2.columns.str.replace(' ', '_')

# Kolone korišćene u step 2
X_all_step2 = df_all_2[model.feature_name_]

# Prvi model: verovatnoća da korisnik ima štetu
proba_claim = final_model.predict_proba(X_all_step1)[:, 1]
mask_pos = proba_claim >= 0.31

# Drugi model: rizik kod onih sa verovatnoćom > prag
X_pos = X_all_step2[mask_pos]
pred_risk = model.predict(X_pos)

# Kombinacija u finalni vektor
final_preds = np.zeros(len(df_copy), dtype=int)
final_preds[mask_pos] = pred_risk + 1

# Konačne predikcije
df_copy['Hurdle_pred'] = final_preds
