from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

knn.fit(X_train, y_train)
train_acc = accuracy_score(y_train, knn.predict(X_train))
test_acc = accuracy_score(y_test, knn.predict(X_test))
cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')

print(f"Train accuracy: {train_acc:.3f}")
print(f"Test accuracy:  {test_acc:.3f}")
print(f"Gap:            {train_acc - test_acc:.3f}")
print(f"CV accuracy:    {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Also worth doing: sweep k values to find the sweet spot
print("\n=== EFFECT OF k ===")
print(f"{'k':>4} | {'train':>6} | {'test':>6} | {'cv mean':>8}")
for k in [1, 3, 5, 7, 10, 15, 20, 30, 50]:
    m = Pipeline([('scaler', StandardScaler()),
                  ('knn', KNeighborsClassifier(n_neighbors=k))])
    m.fit(X_train, y_train)
    tr = accuracy_score(y_train, m.predict(X_train))
    te = accuracy_score(y_test, m.predict(X_test))
    cv = cross_val_score(m, X_train, y_train, cv=5, scoring='accuracy').mean()
    print(f"{k:>4} | {tr:>6.3f} | {te:>6.3f} | {cv:>8.3f}")