# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization


# Load the dataset
file_path = "DATA.xlsx"  # Replace with your dataset file path
data = pd.read_excel(file_path)

# Display the dataset structure
data.head()


# Step 1: Prepare input features (X) and output targets (y)
# Check the number of columns in your DataFrame
num_cols = data.shape[1]

# Adjust slicing to ensure you select at least one column for X
X = data.iloc[:, : num_cols - 10]  # Inputs: All columns except the last 10
# If data has less than 10 columns the previous line will make X to have 0 columns.
# the next line, in that case, will assign all columns but the last one to X.
if X.shape[1] == 0:
    X = data.iloc[:, :-1]

y = data.iloc[
    :, -10:
]  # Outputs: Last 10 columns (NO3, Th, Ca, Mg, Cl, TDS, F, pH, Cr, Fe)
# If data has less than 10 columns the previous line will make y to have 0 columns.
# the next line, in that case, will assign only the last column to y.
if y.shape[1] == 0:
    y = data.iloc[:, -1:]

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Scale the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)


# Build the enhanced ANN Model
model = Sequential()

# Input Layer
model.add(Dense(units=256, activation="relu", input_dim=X_train_scaled.shape[1]))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Hidden Layers
model.add(Dense(units=128, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(units=64, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(units=32, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(units=16, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(units=y_train_scaled.shape[1], activation="linear"))

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

# Train with reasonable batch size
history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=100000,
    batch_size=99999999999,
    validation_split=0.3,
)

# Evaluate and save
test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled)
model.save("water_quality_ann.h5")


plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.savefig("training_validation_loss.png")  # Save the plot as a PNG file


# Load the model
model = load_model("water_quality_ann.h5")
print(model.summary())
file_path = "DATA.xlsx"  # Replace with your dataset file path
data = pd.read_excel(file_path)

# Display the dataset structure
print(data.head())

# Adjust input_count based on model's expected input shape
input_count = 9  # Set to 9 as per model's input requirement
target_count = 10  # Number of output features

# Ensure the dataset has enough columns
if data.shape[1] < input_count:
    raise ValueError(
        f"Dataset must have at least {input_count} columns, but it has {data.shape[1]}"
    )

# Extract input features used during training
historical_data = data.iloc[:, :input_count]

# Prepare the scalers (fit them on the entire dataset)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_X.fit(data.iloc[:, :input_count])
scaler_y.fit(data.iloc[:, :target_count])


# Define the predict_future function
def predict_future(data, scaler_X, scaler_y, input_count, target_count, months):
    predictions = []
    future_data = data.copy()

    for _ in range(months):
        # Scale the input data
        future_scaled = scaler_X.transform(future_data)  # Shape: [1, input_count]

        # Make prediction
        prediction_scaled = model.predict(future_scaled, verbose=0)

        # Inverse transform to original scale
        prediction_original = scaler_y.inverse_transform(prediction_scaled)

        # Append prediction to the list
        predictions.append(prediction_original[0])

        # Prepare the next input using the first 'input_count' predicted features
        future_data = pd.DataFrame(
            [prediction_original[0][:input_count]],
            columns=data.columns[:input_count],
        )

    return predictions


# Get the last known data
last_known_data = historical_data.tail(1)

# Predict the next 2 months
predictions = predict_future(
    last_known_data, scaler_X, scaler_y, input_count, target_count, months=24
)
output_columns = data.columns[:target_count]
# Display Predictions
for i, prediction in enumerate(predictions, start=1):
    print(f"Month {i}:")
    for col_name, value in zip(output_columns, prediction):
        print(f" {col_name}: {abs(value):.2f}")
    print("")  # Add an empty line between months
