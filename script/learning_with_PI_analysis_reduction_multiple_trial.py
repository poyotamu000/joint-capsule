import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.inspection import permutation_importance
from torch.utils.tensorboard import SummaryWriter

# Load csv
df = pd.read_csv("../csv/output_whole_all.csv")

# print number of columns before filtering
print("Number of columns before filtering:", len(df.columns))

# Filter columns based on delta threshold
delta_threshold = 3
adc_columns = [col for col in df.columns if "adc" in col and "stamp" not in col]
deltas = df[adc_columns].max() - df[adc_columns].min()
filtered_adc_columns = deltas[deltas > delta_threshold].index.tolist()
# print number of columns after filtering
print("Number of columns after filtering:", len(filtered_adc_columns))

# Define the output columns
output_columns = [
    "translation_x",
    "translation_y",
    "translation_z",
    "rpy_x",
    "rpy_y",
    "rpy_z",
]

# Prepare the output data
y = df[output_columns].values

# reduction sensor number
reduction_num = 1
reduction_rates = [
    i / float(len(filtered_adc_columns))
    for i in range(0, len(filtered_adc_columns), reduction_num)
]
input_columns = filtered_adc_columns

results = []
training_losses_dict = {}
tidy_rows = []

N_TRIALS = 2
for reduction_rate in reduction_rates:
    print(f"\nReduction rate: {reduction_rate * 100}%")

    for trial in range(1, N_TRIALS + 1):
        print(f"  Trial {trial}/{N_TRIALS}")

        # print number of columns after random removal
        print("Number of columns after random removal:", len(input_columns))
        print(f"input_columns: {input_columns}")

        # Extract filtered columns
        X = df[input_columns].values

        # Split data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42 + trial
        )

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Define model
        class SixAxisNN(nn.Module):
            def __init__(self):
                super(SixAxisNN, self).__init__()
                self.fc1 = nn.Linear(len(input_columns), 128)
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

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss}")

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

        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Loader Length: {len(test_loader)}")
        print(f"Average Test Loss: {avg_test_loss}")

        # convert to numpy array
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)

        predictions_unscaled = scaler_y.inverse_transform(predictions)
        actuals_unscaled = scaler_y.inverse_transform(actuals)

        # calculate metrics
        errors = abs(predictions_unscaled - actuals_unscaled)
        min_error = np.min(errors, axis=0)
        max_error = np.max(errors, axis=0)
        mean_error = np.mean(errors, axis=0)
        std_error = np.std(errors, axis=0)

        roll_mean = float(mean_error[3])
        pitch_mean = float(mean_error[4])
        roll_max = float(max_error[3])
        pitch_max = float(max_error[4])

        tidy_rows.append(
            {
                "Sensor Reduction Ratio": reduction_rate * 100.0,
                "trial_id": trial,
                "Mean Error (Roll)": roll_mean,
                "Mean Error (Pitch)": pitch_mean,
                "Maximum Error (Roll)": roll_max,
                "Maximum Error (Pitch)": pitch_max,
                "Test Loss": avg_test_loss,
            }
        )

        tidy_df = pd.DataFrame(tidy_rows)
        tidy_df.to_csv("trialwise_errors_by_rate.csv", index=False)
        print("Saved trialwise CSV: trialwise_errors_by_rate.csv")

    # Store training losses for this reduction rate
    training_losses_dict[reduction_rate] = train_losses

    np.savez_compressed(
        f"training_losses_rate_{int(reduction_rate*100)}.npz",
        train_losses=np.array(train_losses),
        reduction_rate_percent=reduction_rate * 100.0,
    )

    ##################################################
    ### Calculate permutation importance
    ##################################################

    # Custom wrapper for PyTorch model
    class NNWrapper:
        def __init__(self, model):
            self.model = model

        def fit(self, X, y):
            self.model.train()
            dataset = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
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
            f"Feature: {input_columns[i]}, Importance: {r.importances_mean[i]:.4f} +/- {r.importances_std[i]:.4f}"
        )

    # Save permutation importance to CSV
    filename = f"permutation_importance_{len(input_columns)}.csv"
    perm_importance_df = pd.DataFrame(
        {
            "Feature": input_columns,
            "Importance Mean": r.importances_mean,
            "Importance Std": r.importances_std,
        }
    )
    perm_importance_df.to_csv(filename, index=False)

    # Get the top x features based on permutation importance
    sorted_indices = r.importances_mean.argsort()[::-1]
    top_x_indices = sorted_indices[:reduction_num]
    top_x_features = [input_columns[i] for i in top_x_indices]
    print(f"Top x features based on permutation importance: {top_x_features}")
    print(
        f"Indices of the top x features in the original filtered_adc_columns: {top_x_indices}"
    )

    ##################################################
    # store results for this reduction rate
    results.append(
        {
            "reduction_rate": reduction_rate * 100,
            "input_columns": input_columns,
            "test_loss": avg_test_loss,
            "min_error": min_error,
            "max_error": max_error,
            "mean_error": mean_error,
            "std_error": std_error,
        }
    )

    # Remove top x features from input_columns
    input_columns = [col for col in input_columns if col not in top_x_features]
    print(f"Remaining features after removing top x: {input_columns}")

# Convert results to DataFrame and save as CSV
results_df = pd.DataFrame(results)
results_df.to_csv("reduction_results_multiple_trials.csv", index=False)

# print summarized results
for result in results:
    print(f"Reduction rate: {result['reduction_rate']}%")
    print(f"Input Columns: {result['input_columns']}")
    print(f"Test Loss: {result['test_loss']}")
    print(f"Min Error: {result['min_error']}")
    print(f"Max Error: {result['max_error']}")
    print(f"Mean Error: {result['mean_error']}")
    print(f"Std Error: {result['std_error']}")
    print("\n---------------------------------------")

# Plot training loss curves for each reduction rate
plt.figure(figsize=(12, 8))
for reduction_rate, train_losses in training_losses_dict.items():
    plt.plot(train_losses, label=f"Reduction rate: {reduction_rate * 100:.0f}%")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epochs for Different Reduction Rates")
plt.legend()
plt.grid(True)
plt.savefig("training_loss_curves.png")
plt.show()
