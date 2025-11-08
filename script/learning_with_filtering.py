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
df = pd.read_csv("../csv/filtered_data_roll_over_20deg.csv")

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

# make plot
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
