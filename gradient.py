import xgboost as xgb
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

url= 'titanic.csv'
df = pd.read_csv(url)

df = df.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'])
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
print("Cleaned data preview:")
print(df.head())
print()
print("Any missing values left?", df.isnull().sum().sum())
print()


X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Carve a small validation set out of training data, for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print()
print("=== XGBOOST ===")

xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    early_stopping_rounds=50,
    random_state=42
)

xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"Best iteration (trees actually used): {xgb_model.best_iteration}")

xgb_train_acc = accuracy_score(y_train, xgb_model.predict(X_train))
xgb_test_acc = accuracy_score(y_test, xgb_model.predict(X_test))

print(f"Training accuracy: {xgb_train_acc:.3f}")
print(f"Test accuracy:     {xgb_test_acc:.3f}")
print(f"Gap:               {xgb_train_acc - xgb_test_acc:.3f}")

# Cross-validate to get a more reliable estimate than one split
cv_scores = cross_val_score(
    xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                       subsample=0.8, colsample_bytree=0.8, random_state=42),
    X_train, y_train, cv=5, scoring='accuracy'
)
print(f"CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

print()
print("=== FEATURE IMPORTANCE (XGBoost) ===")
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
print(xgb_importances.sort_values(ascending=False))