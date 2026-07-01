"""
PCA on the Titanic dataset.

Two clearly separated stages:
  1. Feature engineering  -> turn 12 raw columns into a clean numeric matrix
  2. PCA                  -> reduce/decorrelate that numeric matrix

Run: python titanic_pca.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------------
# STAGE 1 — Feature engineering (NOT PCA)
# Raw dataset has 12 columns. Several are unusable as-is:
#   - PassengerId: row index, no signal
#   - Ticket:      free text, mostly noise
#   - Name/Cabin:  free text / mostly missing, BUT contain extractable signal
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- missing values ---
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # --- basic encodings ---
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # --- engineered features (this is where real signal comes from) ---
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]*)\.')
    title_map = {'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master'}
    df['Title'] = df['Title'].map(title_map).fillna('Rare')

    df['Deck'] = df['Cabin'].str[0].fillna('Unknown')

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # --- assemble final numeric feature matrix ---
    base_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                 'FamilySize', 'IsAlone']

    X = pd.concat([
        df[base_cols],
        pd.get_dummies(df['Embarked'], prefix='Emb', drop_first=True),
        pd.get_dummies(df['Title'], prefix='Title', drop_first=True),
        pd.get_dummies(df['Deck'], prefix='Deck', drop_first=True),
    ], axis=1)

    return X


# ---------------------------------------------------------------------------
# STAGE 2 — PCA
# Takes the clean numeric matrix from Stage 1 and finds the directions
# of maximum variance. This is the only part that is actually "PCA".
# ---------------------------------------------------------------------------

def run_pca(X: pd.DataFrame, variance_target: float = 0.90):
    # ALWAYS standardize before PCA (features are on very different scales)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit on all components first to inspect variance
    pca_full = PCA()
    pca_full.fit(X_scaled)

    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    k = int(np.argmax(cumulative >= variance_target) + 1)

    print(f"Input dimensions: {X.shape[1]}")
    print(f"Explained variance ratio (first 5 PCs): {np.round(explained[:5], 3)}")
    print(f"Components needed for {variance_target:.0%} variance: {k}")

    # Refit keeping only k components
    pca_k = PCA(n_components=k)
    X_reduced = pca_k.fit_transform(X_scaled)

    # Loadings: which original features drive each PC
    loadings = pd.DataFrame(
        pca_k.components_.T,
        columns=[f'PC{i+1}' for i in range(k)],
        index=X.columns,
    )

    return X_reduced, pca_k, loadings, scaler



if __name__ == '__main__':
    df = pd.read_csv('titanic.csv')  # adjust path as needed

    X = build_features(df)
    y = df['Survived']

    print(f"Raw columns: {df.shape[1]}  ->  Engineered features: {X.shape[1]}\n")

    X_reduced, pca_model, loadings, scaler = run_pca(X, variance_target=0.90)

    print("\nTop feature loadings for PC1 and PC2:")
    print(loadings[['PC1', 'PC2']].reindex(
        loadings['PC1'].abs().sort_values(ascending=False).index
    ).head(8).round(3))

    