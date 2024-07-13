# Fraud Detection in Financial Transactions

## Description

This project aims to detect fraudulent transactions in a financial dataset using machine learning techniques. The project involves data preprocessing, feature engineering, exploratory data analysis (EDA), model selection, training, and evaluation.

## Technologies Used

- **Python**: The programming language used for implementation.
- **Pandas**: Used for data manipulation and analysis.
- **NumPy**: Used for numerical computations.
- **Matplotlib**: Used for data visualization.
- **Seaborn**: Used for statistical data visualization.
- **Scikit-learn**: Used for machine learning model building and evaluation.
- **Imbalanced-learn**: Used for handling imbalanced datasets.
- **TensorFlow (Keras)**: Used for building and training neural network models.

## Key Highlights

### Data Preprocessing

1. **Loading the dataset**:
    ```python
    import pandas as pd
    import numpy as np
    data = pd.read_csv('creditcard.csv')
    ```

2. **Checking for missing values**:
    ```python
    print(data.isnull().sum())
    ```

3. **Handling missing values** (if any):
    ```python
    # In this dataset, there are no missing values, but if there were, you could use:
    # data.fillna(data.mean(), inplace=True)
    ```

4. **Checking the balance of the dataset**:
    ```python
    print(data['Class'].value_counts())
    ```

5. **Balancing the dataset using under-sampling**:
    ```python
    from imblearn.under_sampling import RandomUnderSampler
    undersample = RandomUnderSampler(sampling_strategy=1)
    X = data.drop('Class', axis=1)
    y = data['Class']
    X_over, y_over = undersample.fit_resample(X, y)
    balanced_data = pd.concat([X_over, y_over], axis=1)
    ```

### Exploratory Data Analysis (EDA)

1. **Visualizing the distribution of transaction amounts**:
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10,5))
    sns.histplot(data['Amount'], bins=50, kde=True)
    plt.title('Transaction Amount Distribution')
    plt.show()
    ```

2. **Visualizing the correlation matrix**:
    ```python
    plt.figure(figsize=(20,10))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    ```

3. **Visualizing the count of fraudulent vs non-fraudulent transactions**:
    ```python
    plt.figure(figsize=(7,5))
    sns.countplot(x='Class', data=balanced_data)
    plt.title('Fraudulent vs Non-Fraudulent Transactions')
    plt.show()
    ```

### Model Training and Evaluation

1. **Splitting the data into training and testing sets**:
    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.3, random_state=42)
    ```

2. **Standardizing the data**:
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    ```

3. **Model 1: Random Forest**:
    ```python
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    ```

4. **Model 2: Neural Network**:
    ```python
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    nn_model = Sequential()
    nn_model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
    nn_model.add(Dense(16, activation='relu'))
    nn_model.add(Dense(1, activation='sigmoid'))

    nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    nn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    nn_preds = (nn_model.predict(X_test) > 0.5).astype("int32")
    ```

5. **Evaluation Metrics**:
    - **Random Forest**:
        ```python
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        rf_accuracy = accuracy_score(y_test, rf_preds)
        rf_precision = precision_score(y_test, rf_preds)
        rf_recall = recall_score(y_test, rf_preds)
        rf_f1 = f1_score(y_test, rf_preds)

        print(f'Random Forest - Accuracy: {rf_accuracy}, Precision: {rf_precision}, Recall: {rf_recall}, F1 Score: {rf_f1}')
        ```

    - **Neural Network**:
        ```python
        nn_accuracy = accuracy_score(y_test, nn_preds)
        nn_precision = precision_score(y_test, nn_preds)
        nn_recall = recall_score(y_test, nn_preds)
        nn_f1 = f1_score(y_test, nn_preds)

        print(f'Neural Network - Accuracy: {nn_accuracy}, Precision: {nn_precision}, Recall: {nn_recall}, F1 Score: {nn_f1}')
        ```

## Results

Evaluation metrics for the models will be displayed after running the evaluation script. The results include accuracy, precision, recall, and F1-score.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
