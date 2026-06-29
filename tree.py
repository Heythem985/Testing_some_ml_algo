import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score



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
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")


print()
print("=== RANDOM FOREST vs SINGLE TREE ===")

# Best single tree from your depth sweep (pick whichever depth had the best test_acc)
best_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
best_tree.fit(X_train, y_train)
tree_test_acc = accuracy_score(y_test, best_tree.predict(X_test))

# Random Forest baseline
rf = RandomForestClassifier(
    n_estimators=300,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    oob_score=True
)
rf.fit(X_train, y_train)

rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))

print(f"Single tree (max_depth=4) test accuracy: {tree_test_acc:.3f}")
print(f"Random Forest train accuracy:            {rf_train_acc:.3f}")
print(f"Random Forest test accuracy:             {rf_test_acc:.3f}")
print(f"Random Forest OOB score:                 {rf.oob_score_:.3f}")
print(f"Random Forest train/test gap:            {rf_train_acc - rf_test_acc:.3f}")

# Cross-validate so this isn't just one lucky train/test split
print()
print("=== 5-FOLD CROSS-VALIDATION ===")
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV accuracy per fold: {[round(s, 3) for s in cv_scores]}")
print(f"CV mean ± std:        {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Feature importance
print()
print("=== FEATURE IMPORTANCE ===")
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
print(importances.sort_values(ascending=False))
