import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

data = pd.read_csv("drug200.csv")

categorical_cols = ['Sex', 'BP', 'Cholesterol']
data_encoded = data.copy()
le = LabelEncoder()
for col in categorical_cols:
    data_encoded[col] = le.fit_transform(data_encoded[col])

X = data_encoded.drop('Drug', axis=1)
y = data_encoded['Drug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}
grid_tree = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_tree, cv=5)
grid_tree.fit(X_train, y_train)

param_grid_lr = {
    'solver': ['liblinear', 'lbfgs'],
    'C': [0.01, 0.1, 1, 10]
}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5)
grid_lr.fit(X_train, y_train)

y_pred_tree = grid_tree.predict(X_test)
y_pred_lr = grid_lr.predict(X_test)

print("Средняя точность дерева решений:", accuracy_score(y_test, y_pred_tree))
print("Отчёт классификации для дерева решений:\n", classification_report(y_test, y_pred_tree))

print("Средняя точность логистической регрессии:", accuracy_score(y_test, y_pred_lr))
print("Отчёт классификации для логистической регрессии:\n", classification_report(y_test, y_pred_lr))

ConfusionMatrixDisplay.from_estimator(grid_tree, X_test, y_test)
plt.title("Матрица ошибок – дерево решений")
plt.show()

ConfusionMatrixDisplay.from_estimator(grid_lr, X_test, y_test)
plt.title("Матрица ошибок – логистическая регрессия")
plt.show()

depths = list(range(1, 21))
scores = []
for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    score = cross_val_score(clf, X_train, y_train, cv=5).mean()
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(depths, scores, marker='o')
plt.xlabel('Глубина дерева решений')
plt.ylabel('Средняя точность (кросс-валидация)')
plt.title('Влияние глубины дерева на точность модели')
plt.grid(True)
plt.show()

C_values = [0.01, 0.1, 1, 10, 100]
scores_lr = []
for c in C_values:
    model = LogisticRegression(C=c, max_iter=1000, solver='liblinear')
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    scores_lr.append(score)

plt.figure(figsize=(8, 5))
plt.plot(C_values, scores_lr, marker='s')
plt.xscale('log')
plt.xlabel('Значение C (логарифмическая шкала)')
plt.ylabel('Средняя точность (кросс-валидация)')
plt.title('Влияние параметра регуляризации C на логистическую регрессию')
plt.grid(True)
plt.show()

accuracy_tree = accuracy_score(y_test, y_pred_tree)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

plt.figure(figsize=(6, 4))
plt.bar(['Дерево решений', 'Логистическая регрессия'], [accuracy_tree, accuracy_lr],
        color=['skyblue', 'lightgreen'])
plt.ylabel('Точность на тестовой выборке')
plt.title('Сравнение точности моделей')
plt.ylim(0.8, 1.0)
plt.grid(axis='y')
plt.show()