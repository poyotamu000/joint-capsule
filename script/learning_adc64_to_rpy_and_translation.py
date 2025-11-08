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
df = pd.read_csv("../csv/output_whole_all.csv")

# Extract columns
input_columns = [f"adc{i}_data{j}" for i in range(4) for j in range(16)]
output_columns = [
    "translation_x",
    "translation_y",
    "translation_z",
    "rpy_x",
    "rpy_y",
    "rpy_z",
]
X = df[input_columns].values
y = df[output_columns].values

# Split data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# transform to tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

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
        self.fc1 = nn.Linear(64, 128)
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
results = []

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
predictions = np.array(predictions)
actuals = np.array(actuals)
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


# convert to numpy array
predictions2 = np.concatenate(predictions, axis=0)
actuals2 = np.concatenate(actuals, axis=0)

predictions_unscaled2 = scaler_y.inverse_transform(predictions2)
actuals_unscaled2 = scaler_y.inverse_transform(actuals2)

# calculate metrics
errors = abs(predictions_unscaled2 - actuals_unscaled2)
min_error = np.min(errors, axis=0)
max_error = np.max(errors, axis=0)
mean_error = np.mean(errors, axis=0)
std_error = np.std(errors, axis=0)

# print metrics
print(f"Min Error: {min_error}")
print(f"Max Error: {max_error}")
print(f"Mean Error: {mean_error}")
print(f"Standard Deviation of Error: {std_error}")

results.append(
    {
        "test_loss": test_loss / len(test_loader),
        "min_error": min_error,
        "max_error": max_error,
        "mean_error": mean_error,
        "std_error": std_error,
    }
)

results_df = pd.DataFrame(results)
results_df.to_csv("learning_adc64_to_rpy_and_translation.csv", index=False)


# make plot
titles = [
    "x translation",
    "y translation",
    "z translation",
    "roll angle",
    "pitch angle",
    "yaw angle",
]
fig, axes = plt.subplots(2, 3, figsize=(18, 6))

for i in range(6):
    row = i // 3
    col = i % 3
    for j in range(len(predictions)):
        axes[row, col].plot(
            actuals_unscaled[j][:, i],
            label=f"True {titles[i]}" if j == 0 else "",
            alpha=0.5,
        )
        axes[row, col].plot(
            predictions_unscaled[j][:, i],
            label=f"Predicted {titles[i]}" if j == 0 else "",
            linestyle="dashed",
            alpha=0.5,
        )
    axes[row, col].set_title(titles[i])
    if i == 0:
        axes[row, col].legend()

plt.tight_layout()
plt.show()


# make plot for each sample
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
    # plt.suptitle(f"Sample {sample_idx + 1}")
    plt.subplots_adjust(top=0.95)
    plt.show()
