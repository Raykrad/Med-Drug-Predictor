import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

data = pd.read_csv("drug200.csv")
le_dict = {}
categorical_cols = ['Sex', 'BP', 'Cholesterol']
data_encoded = data.copy()

for col in categorical_cols:
    le = LabelEncoder()
    data_encoded[col] = le.fit_transform(data_encoded[col])
    le_dict[col] = le 

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

joblib.dump(grid_tree.best_estimator_, 'saved_model.pkl')
joblib.dump(le_dict, 'label_encoders.pkl')

print("Модель обучена и сохранена в 'saved_model.pkl'")
