import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the model
model = load_model("water_quality_ann_enhanced.h5")
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
