from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
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

print()
print("=== SVM ===")

# StandardScaler is mandatory here — SVM is distance/margin-based,
# so Fare (0-512 range) would dominate Age (0-80) and dummy columns (0/1)
# purely due to scale, not actual predictive power.

# Use a Pipeline so scaling is refit correctly inside each CV fold —
# fitting the scaler on the full training set before CV would leak
# information from each fold's held-out portion into the scaling stats.

svm_linear = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='linear', C=1.0, random_state=42))
])

svm_rbf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])

for name, model in [('Linear SVM', svm_linear), ('RBF SVM', svm_rbf)]:
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    print(f"\n{name}")
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy:  {test_acc:.3f}")
    print(f"  Gap:            {train_acc - test_acc:.3f}")
    print(f"  CV accuracy:    {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")