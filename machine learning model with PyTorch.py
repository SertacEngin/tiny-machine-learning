import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# Load the data with specified delimiter
data = pd.read_csv('Data.csv', delimiter=';')

# Split the data into features (X) and target variables (y)
X = data[['Power Consumption', 'Time']].values
y_object = data['Object'].astype(int).values  # Only 'Object' column for binary classification
y_path = pd.get_dummies(data['Path']).values  # Only 'Path' column for multi-class classification

# Split the data into training and testing sets
X_train, X_test, y_object_train, y_object_test, y_path_train, y_path_test = train_test_split(
    X, y_object, y_path, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_object_train_tensor = torch.tensor(y_object_train, dtype=torch.float32)
y_object_test_tensor = torch.tensor(y_object_test, dtype=torch.float32)
y_path_train_tensor = torch.tensor(y_path_train, dtype=torch.float32)
y_path_test_tensor = torch.tensor(y_path_test, dtype=torch.float32)

# Create DataLoader
train_dataset_object = TensorDataset(X_train_tensor, y_object_train_tensor)
test_dataset_object = TensorDataset(X_test_tensor, y_object_test_tensor)
train_loader_object = DataLoader(train_dataset_object, batch_size=32, shuffle=True)
test_loader_object = DataLoader(test_dataset_object, batch_size=32, shuffle=False)

train_dataset_path = TensorDataset(X_train_tensor, y_path_train_tensor)
test_dataset_path = TensorDataset(X_test_tensor, y_path_test_tensor)
train_loader_path = DataLoader(train_dataset_path, batch_size=32, shuffle=True)
test_loader_path = DataLoader(test_dataset_path, batch_size=32, shuffle=False)

# Define the neural network model for object prediction
class ObjectPredictionModel(nn.Module):
    def __init__(self):
        super(ObjectPredictionModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Define the neural network model for path prediction
class PathPredictionModel(nn.Module):
    def __init__(self):
        super(PathPredictionModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Instantiate models
model_object = ObjectPredictionModel()
model_path = PathPredictionModel()

# Define loss and optimizer
criterion_object = nn.BCELoss()
optimizer_object = optim.Adam(model_object.parameters(), lr=0.001)

criterion_path = nn.CrossEntropyLoss()
optimizer_path = optim.Adam(model_path.parameters(), lr=0.001)

# Train the model for object prediction
num_epochs = 50
for epoch in range(num_epochs):
    model_object.train()
    for X_batch, y_batch in train_loader_object:
        optimizer_object.zero_grad()
        outputs = model_object(X_batch).squeeze()
        loss = criterion_object(outputs, y_batch)
        loss.backward()
        optimizer_object.step()

# Evaluate the model for object prediction
model_object.eval()
with torch.no_grad():
    outputs = model_object(X_test_tensor).squeeze()
    loss_object = criterion_object(outputs, y_object_test_tensor)
    accuracy_object = accuracy_score(y_object_test, (outputs > 0.5).int().numpy())

print("Object Prediction Test Loss:", loss_object.item())
print("Object Prediction Test Accuracy:", accuracy_object)

# Train the model for path prediction
for epoch in range(num_epochs):
    model_path.train()
    for X_batch, y_batch in train_loader_path:
        optimizer_path.zero_grad()
        outputs = model_path(X_batch)
        loss = criterion_path(outputs, y_batch)
        loss.backward()
        optimizer_path.step()

# Evaluate the model for path prediction
model_path.eval()
with torch.no_grad():
    outputs = model_path(X_test_tensor)
    loss_path = criterion_path(outputs, y_path_test_tensor)
    accuracy_path = accuracy_score(y_path_test.argmax(axis=1), outputs.argmax(axis=1).numpy())

print("Path Prediction Test Loss:", loss_path.item())
print("Path Prediction Test Accuracy:", accuracy_path)

# Make predictions
new_data = pd.DataFrame({'Power Consumption': [15119], 'Time': [26.64]})
new_data_scaled = scaler.transform(new_data)
new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

# Object prediction
with torch.no_grad():
    object_prediction = model_object(new_data_tensor).item()
print("Object Prediction:", int(object_prediction > 0.5))

# Path prediction
with torch.no_grad():
    path_prediction_probabilities = model_path(new_data_tensor)
    path_prediction = [23, 25, 27, 28][path_prediction_probabilities.argmax().item()]
print("Path Prediction:", path_prediction)
