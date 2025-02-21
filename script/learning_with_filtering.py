import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load csv
df = pd.read_csv("../csv/py_filtered_data_over_70.csv")

# print number of columns before filtering
print("Number of columns before filtering:", len(df.columns))

# Filter columns based on delta threshold
delta_threshold = 3
adc_columns = [col for col in df.columns if "adc" in col and "stamp" not in col]
deltas = df[adc_columns].max() - df[adc_columns].min()
filtered_adc_columns = deltas[deltas > delta_threshold].index.tolist()

# print exclude columns
exclude_columns = [col for col in adc_columns if col not in filtered_adc_columns]
print("Exclude columns:", exclude_columns)

# print number of columns after filtering
print("Number of columns after filtering:", len(filtered_adc_columns))
print("Filtered columns:", filtered_adc_columns)

# # Extract only the sensors in the area close to the bone.
# data_indices = ["data0", "data1", "data2", "data9", "data10", "data11"]
# selected_columns = [
#     col
#     for col in filtered_adc_columns
#     if any(col.endswith(index) for index in data_indices)
# ]
# print("Selected columns:", selected_columns)
# filtered_adc_columns = selected_columns
# print("Filtered columns:", filtered_adc_columns)

# Extract filtered columns
output_columns = [
    "translation_x",
    "translation_y",
    "translation_z",
    "rpy_x",
    "rpy_y",
    "rpy_z",
]
X = df[filtered_adc_columns].values
y = df[output_columns].values

# Split data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)


##################################################
### Calculate mutual information
##################################################
from sklearn.feature_selection import mutual_info_regression

# We need to calculate mutual information for each target variable
# That is, we need to calculate it 6 times for 6-dimensional target variables
mi_scores_list = []
for i in range(y_train.shape[1]):
    mi_scores = mutual_info_regression(X_train, y_train[:, i])
    mi_scores_list.append(mi_scores)
    score_feature_tuples = [
        (score, feature) for score, feature in zip(mi_scores, filtered_adc_columns)
    ]
    score_feature_tuples_sorted = sorted(
        score_feature_tuples, key=lambda x: x[0], reverse=True
    )
    print(f"Features sorted by mutual information score (high to low) for target {i}:")
    for score, feature in score_feature_tuples_sorted:
        print(f"{feature}: {score:.2f}")

# Save mutual information scores to CSV
mi_scores_df = pd.DataFrame(
    mi_scores_list, columns=filtered_adc_columns, index=output_columns
)
mi_scores_df.to_csv("mutual_information_scores.csv", index=True)
##################################################


train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define model
class SixAxisNN(nn.Module):
    def __init__(self):
        super(SixAxisNN, self).__init__()
        self.fc1 = nn.Linear(len(filtered_adc_columns), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


model = SixAxisNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# train model
num_epochs = 100
train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Save training loss to CSV
loss_df = pd.DataFrame(train_losses, columns=["Training Loss"])
loss_df.to_csv("training_loss.csv", index=False)

# plot loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.show()


# evaluate model
model.eval()
test_loss = 0.0
predictions = []
actuals = []


with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        predictions.append(outputs.numpy())
        actuals.append(targets.numpy())

print(f"Test Loader Length: {len(test_loader)}")
print(f"Test Loss: {test_loss/len(test_loader)}")


# convert to numpy array
predictions = np.array(predictions, dtype=object)
actuals = np.array(actuals, dtype=object)
# check array shape
print(f"Predictions shape: {predictions.shape}")
print(f"Actuals shape: {actuals.shape}")
print(f"Actuals[0] shape: {actuals[0].shape}")


# inverse transform
predictions_unscaled = np.zeros_like(predictions)
actuals_unscaled = np.zeros_like(actuals)

for i in range(len(predictions)):
    predictions_unscaled[i] = scaler_y.inverse_transform(
        predictions[i].reshape(-1, 6)
    ).reshape(predictions[i].shape)
    actuals_unscaled[i] = scaler_y.inverse_transform(actuals[i].reshape(-1, 6)).reshape(
        actuals[i].shape
    )

# Save predictions and actuals to CSV
# predictions_df = pd.DataFrame(
#     predictions_unscaled.reshape(-1, 6), columns=output_columns
# )
# actuals_df = pd.DataFrame(actuals_unscaled.reshape(-1, 6), columns=output_columns)
# predictions_df.to_csv("predictions.csv", index=False)
# actuals_df.to_csv("actuals.csv", index=False)


##################################################
### Calculate shap
##################################################

import shap

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
explainer = shap.DeepExplainer(model, X_train_tensor)
shap_values = explainer.shap_values(X_test_tensor, check_additivity=False)
shap.summary_plot(shap_values, X_test, feature_names=filtered_adc_columns)
for i in range(y_train.shape[1]):
    shap.summary_plot(
        shap_values[i], X_test, feature_names=filtered_adc_columns, plot_type="bar"
    )

# Save SHAP values to CSV
for i in range(len(shap_values)):
    shap_df = pd.DataFrame(shap_values[i], columns=filtered_adc_columns)
    shap_df.to_csv(f"shap_values_target_{i}.csv", index=False)

##################################################
### plot
##################################################

# make plot
# fig, axes = plt.subplots(2, 3, figsize=(18, 6))

# axes[0, 0].plot(actuals_unscaled[0][:, 0], label="True translation_x")
# axes[0, 0].plot(
#     predictions_unscaled[0][:, 0], label="Predicted translation_x", linestyle="dashed"
# )
# axes[0, 0].set_title("translation_x")
# axes[0, 0].legend()

# axes[0, 1].plot(actuals_unscaled[0][:, 1], label="True translation_y")
# axes[0, 1].plot(
#     predictions_unscaled[0][:, 1], label="Predicted translation_y", linestyle="dashed"
# )
# axes[0, 1].set_title("translation_y")
# axes[0, 1].legend()

# axes[0, 2].plot(actuals_unscaled[0][:, 2], label="True translation_z")
# axes[0, 2].plot(
#     predictions_unscaled[0][:, 2], label="Predicted translation_z", linestyle="dashed"
# )
# axes[0, 2].set_title("translation_z")
# axes[0, 2].legend()

# axes[1, 0].plot(actuals_unscaled[0][:, 3], label="True rpy_x")
# axes[1, 0].plot(
#     predictions_unscaled[0][:, 3], label="Predicted rpy_x", linestyle="dashed"
# )
# axes[1, 0].set_title("rpy_x")
# axes[1, 0].legend()

# axes[1, 1].plot(actuals_unscaled[0][:, 4], label="True rpy_y")
# axes[1, 1].plot(
#     predictions_unscaled[0][:, 4], label="Predicted rpy_y", linestyle="dashed"
# )
# axes[1, 1].set_title("rpy_y")
# axes[1, 1].legend()

# axes[1, 2].plot(actuals_unscaled[0][:, 5], label="True rpy_z")
# axes[1, 2].plot(
#     predictions_unscaled[0][:, 5], label="Predicted rpy_z", linestyle="dashed"
# )
# axes[1, 2].set_title("rpy_z")
# axes[1, 2].legend()

# plt.show()


# make plot ver2 (all test loader once
# titles = ["translation_x", "translation_y", "translation_z", "rpy_x", "rpy_y", "rpy_z"]
# fig, axes = plt.subplots(2, 3, figsize=(18, 6))

# for i in range(6):
#     row = i // 3
#     col = i % 3
#     for j in range(len(predictions)):
#         axes[row, col].plot(
#             actuals_unscaled[j][:, i],
#             label=f"True {titles[i]}" if j == 0 else "",
#             alpha=0.5,
#         )
#         axes[row, col].plot(
#             predictions_unscaled[j][:, i],
#             label=f"Predicted {titles[i]}" if j == 0 else "",
#             linestyle="dashed",
#             alpha=0.5,
#         )
#     axes[row, col].set_title(titles[i])
#     if i == 0:
#         axes[row, col].legend()

# plt.tight_layout()
# plt.show()


# make plot ver3 (Each test loader)
titles = ["translation_x", "translation_y", "translation_z", "rpy_x", "rpy_y", "rpy_z"]

for sample_idx in range(len(predictions_unscaled)):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for i in range(6):
        row = i // 3
        col = i % 3
        axes[row, col].plot(actuals_unscaled[sample_idx][:, i], label="True", alpha=0.5)
        axes[row, col].plot(
            predictions_unscaled[sample_idx][:, i],
            label="Predicted",
            linestyle="dashed",
            alpha=0.5,
        )
        axes[row, col].set_title(titles[i])
        axes[row, col].legend()

    plt.tight_layout()
    plt.suptitle(f"Sample {sample_idx + 1}")
    plt.subplots_adjust(top=0.95)
    plt.show()


##################################################
### Calculate permutation importance
##################################################
from sklearn.inspection import permutation_importance


# Custom wrapper for PyTorch model
class NNWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.train()
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(100):  # You can adjust the number of epochs
            running_loss = 0.0
            for inputs, targets in loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(inputs)
        return outputs.numpy()

    def score(self, X, y):
        predictions = self.predict(X)
        return -mean_squared_error(y, predictions)


# Wrap the trained PyTorch model
nn_model = NNWrapper(model)

# Calculate permutation importance
r = permutation_importance(
    nn_model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_squared_error",
)

# Print permutation importances
for i in range(len(r.importances_mean)):
    print(
        f"Feature: {filtered_adc_columns[i]}, Importance: {r.importances_mean[i]:.4f} +/- {r.importances_std[i]:.4f}"
    )

# Save permutation importance to CSV
perm_importance_df = pd.DataFrame(
    {
        "Feature": filtered_adc_columns,
        "Importance Mean": r.importances_mean,
        "Importance Std": r.importances_std,
    }
)
perm_importance_df.to_csv("permutation_importance.csv", index=False)

# Plot permutation importances
plt.figure(figsize=(12, 6))
plt.bar(range(len(filtered_adc_columns)), r.importances_mean, yerr=r.importances_std)
plt.xticks(range(len(filtered_adc_columns)), filtered_adc_columns, rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Permutation Importance of each feature")
plt.tight_layout()
plt.show()
