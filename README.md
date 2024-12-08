# AquaForecaster: Water Quality Forecasting Using Machine Learning

## **Objective**

The primary goal of this project is to develop a predictive model that forecasts future water quality metrics over a specified time horizon (e.g., 12 or 24 months). By leveraging historical water quality data and a trained Artificial Neural Network (ANN), the project aims to provide actionable insights for proactive water resource management.

## **Key Features**

1. **Multi-Output Forecasting**

    - Predicts multiple water quality parameters simultaneously, such as **Nitrate (NO3)**, **Thorium (Th)**, **Calcium (Ca)**, **Magnesium (Mg)**, **Chloride (Cl)**, **Total Dissolved Solids (TDS)**, **Fluoride (F)**, **pH**, **Chromium (Cr)**, and **Iron (Fe)**.

2. **Time-Series Predictions**

    - Generates forecasts for water quality over several months into the future based on the most recent data, making it a time-series forecasting project.

3. **Scalable Input Handling**

    - Dynamically adapts to the number of features in the dataset, making the solution flexible for various water quality monitoring systems.

4. **Machine Learning Model**

    - Utilizes a pre-trained **Artificial Neural Network (ANN)**, trained on historical water quality data to capture complex relationships between variables.

5. **Data Normalization**

    - Applies **StandardScaler** for normalization, ensuring that features are scaled consistently for accurate predictions.

6. **Self-Adaptive Forecasting**
    - Iteratively uses predicted values as inputs for subsequent forecasts, enabling it to predict a sequence of future time steps.

## **Applications**

-   **Water Resource Management**: Helps water authorities and environmental agencies monitor and predict water quality to ensure compliance with safety standards.
-   **Early Warning Systems**: Enables proactive measures to address potential contamination or degradation in water quality.
-   **Sustainability Efforts**: Supports planning and sustainability strategies for clean water access.

## **Workflow**

1. **Data Preparation**

    - Load historical water quality data from `Data/DATA.xlsx`.
    - Standardize the data for consistency using `train.py`.

2. **Model Training and Loading**

    - Train an ANN model using `train.py` (if not already trained) or load an existing pre-trained model `model.h5` or `water_quality_ann_enhanced.h5`.

3. **Prediction Process**

    - Use the latest known data as the initial input.
    - Predict water quality for the desired future period (e.g., 12 or 24 months) using `predict.py`.
    - Continuously use predicted values as inputs for subsequent months.

4. **Result Analysis**
    - Transform predictions back to their original scale.
    - Visualize or tabulate the results for easy interpretation using `output.md`.

## **Technology Stack**

-   **Programming Language**: Python
-   **Libraries**:
    -   `TensorFlow/Keras`: For ANN model training and prediction.
    -   `scikit-learn`: For data preprocessing and normalization.
    -   `NumPy`: For numerical operations.
-   **Tools**: Jupyter Notebook (`NoteBook.ipynb`) or IDE for development and analysis.

## **Future Enhancements**

1. **Improved Accuracy**: Experiment with advanced machine learning models such as LSTM or Transformer-based models tailored for time-series data.
2. **Feature Expansion**: Include external variables like weather data or upstream water flow metrics for enhanced predictive capability.
3. **Visualization**: Build a dashboard for real-time monitoring and visualization of predictions.
4. **Integration**: Incorporate the model into IoT-enabled water monitoring systems.

## **Impact**

This project contributes to sustainable water resource management by leveraging modern machine learning techniques to predict water quality trends. It empowers stakeholders to make data-driven decisions for safeguarding public health and ensuring the availability of clean water.

## **Data**

-   `Data/peenya (2).docx`: Documentation related to the dataset.
-   `Data/peenya results - 2021.xls`: Excel file containing water quality results.
-   `DATA.xlsx`: Master dataset for training and prediction.

## **Models**

-   `model.h5`: Pre-trained ANN model.
-   `water_quality_ann_enhanced.h5`: Enhanced ANN model for improved predictions.

## **Scripts**

-   `train.py`: Script for training the ANN model.
-   `predict.py`: Script for making predictions using the trained model.

## **Notebooks**

-   `NoteBook.ipynb`: Jupyter Notebook for exploratory analysis and development.

## **Outputs**

-   `output.md`: Markdown file containing analysis results.
-   `nohup.out`: Output log from running long processes.

## **Papers**

-   Contains research papers and references related to the project.

For more details on each component, please refer to the respective files in the repository.
