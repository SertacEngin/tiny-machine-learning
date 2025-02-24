import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import graphviz

# Load the data with specified delimiter
data = pd.read_csv('Data.csv', delimiter=';')

# Split the data into features (X) and target variables (y)
X = data[['Power Consumption', 'Time']].values
y_object = data['Object'].astype(int).values  # Only 'Object' column for binary classification
y_path = data['Path'].values  # Only 'Path' column for multi-class classification

# Split the data into training and testing sets
X_train, X_test, y_object_train, y_object_test, y_path_train, y_path_test = train_test_split(
    X, y_object, y_path, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree for object prediction (binary classification)
object_tree = DecisionTreeClassifier(random_state=42)
object_tree.fit(X_train_scaled, y_object_train)

# Evaluate the object prediction model
y_object_pred = object_tree.predict(X_test_scaled)
accuracy_object = accuracy_score(y_object_test, y_object_pred)
print("Object Prediction Test Accuracy:", accuracy_object)

# Visualize the decision tree for object prediction
object_tree_dot = export_graphviz(object_tree, out_file=None,
                                  feature_names=['Power Consumption', 'Time'],
                                  class_names=['0', '1'], filled=True, rounded=True,
                                  special_characters=True)
graph_object = graphviz.Source(object_tree_dot)
graph_object.render("object_tree")

# Decision Tree for path prediction (multi-class classification)
path_tree = DecisionTreeClassifier(random_state=42)
path_tree.fit(X_train_scaled, y_path_train)

# Evaluate the path prediction model
y_path_pred = path_tree.predict(X_test_scaled)
accuracy_path = accuracy_score(y_path_test, y_path_pred)
print("Path Prediction Test Accuracy:", accuracy_path)

# Visualize the decision tree for path prediction
path_tree_dot = export_graphviz(path_tree, out_file=None,
                                feature_names=['Power Consumption', 'Time'],
                                class_names=path_tree.classes_.astype(str), filled=True,
                                rounded=True, special_characters=True)
graph_path = graphviz.Source(path_tree_dot)
graph_path.render("path_tree")

# Make predictions
new_data = pd.DataFrame({'Power Consumption': [15119], 'Time': [26.64]})
new_data_scaled = scaler.transform(new_data)

# Object prediction
object_prediction = object_tree.predict(new_data_scaled)[0]
print("Object Prediction:", object_prediction)

# Path prediction
path_prediction = path_tree.predict(new_data_scaled)[0]
print("Path Prediction:", path_prediction)

# Feature importance
print("Object Tree Feature Importances:", object_tree.feature_importances_)
print("Path Tree Feature Importances:", path_tree.feature_importances_)

# Cross-validation
cv_scores_object = cross_val_score(object_tree, X_train_scaled, y_object_train, cv=5)
cv_scores_path = cross_val_score(path_tree, X_train_scaled, y_path_train, cv=5)

print("Object Tree Cross-Validation Scores:", cv_scores_object)
print("Path Tree Cross-Validation Scores:", cv_scores_path)
print("Object Tree Average CV Score:", cv_scores_object.mean())
print("Path Tree Average CV Score:", cv_scores_path.mean())
