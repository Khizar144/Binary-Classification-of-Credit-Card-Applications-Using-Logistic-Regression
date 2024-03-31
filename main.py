import pandas as pd
import numpy as np
#
# #read the data set
# credit_card_data=pd.read_csv("D:/AI Assignment/Credit_card.csv")
# pd.set_option('display.max_columns', None)  # To display all columns
# pd.set_option('display.max_rows', None)  # To display all rows
# # label dataset
# label_data=pd.read_csv("D:/AI Assignment/Credit_card_label.csv")
# #marge the dataset
# result_df = pd.merge(credit_card_data, label_data, on='Ind_ID')
# # print(label_data.head())
# # Save the merge dataset
# # result_df.to_csv("D:/AI Assignment/merged_dataset.csv",index=False)
# print(result_df.head())
# # print(result_df.isnull().sum())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def predict_proba(self, X):
        z = np.dot(X, self.theta).flatten()  # Compute dot product and flatten the result
        return self.sigmoid(z)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  # Clip values to avoid overflow
        return 1 / (1 + np.exp(-z.astype(float)))  # Ensure z is of type float

    def fit(self, X, y):
        self.theta = np.zeros((X.shape[1], 1))  # Initialize theta as a column vector
        m = len(y)
        y = y.reset_index(drop=True)  # Reset index of y to ensure consistency
        for _ in range(self.max_iter):
            old_theta = self.theta.copy()
            random_index = np.random.randint(m)
            xi = np.array(X.iloc[random_index]).reshape(1, -1)  # Convert xi to numpy array
            xi = xi.astype(float)  # Ensure xi is of type float
            yi = y[random_index]
            z = np.dot(xi, self.theta).flatten()  # Dot product operation
            h = self.sigmoid(z)
            gradient = np.dot(xi.T, (h - yi))
            self.theta -= self.learning_rate * gradient[:, np.newaxis]
            if np.linalg.norm(self.theta - old_theta) < self.tol:
                break

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


class LogisticRegressionBatchGD:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def predict_proba(self, X):
        z = np.dot(X, self.theta).flatten()  # Compute dot product and flatten the result
        return self.sigmoid(z)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  # Clip values to avoid overflow
        return 1 / (1 + np.exp(-z.astype(float)))  # Ensure z is of type float

    def fit(self, X, y):
        self.theta = np.zeros((X.shape[1], 1))  # Initialize theta as a column vector
        m = len(y)
        y = y.reset_index(drop=True)  # Reset index of y to ensure consistency
        for _ in range(self.max_iter):
            old_theta = self.theta.copy()
            z = np.dot(X, self.theta).flatten()
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / m
            gradient = gradient.astype(float)  # Ensure gradient is of type float
            self.theta -= self.learning_rate * gradient[:, np.newaxis]
            if np.linalg.norm(self.theta - old_theta) < self.tol:
                break

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


def plot_accuracy(X, y, learning_rate=0.01):
    max_iters = [100, 500, 1000, 1500, 2000]  # Vary maximum iterations
    accuracies_sgd = []
    accuracies_bgd = []

    for max_iter in max_iters:
        # Stochastic Gradient Descent
        log_reg_sgd = LogisticRegressionSGD(learning_rate=learning_rate, max_iter=max_iter)
        log_reg_sgd.fit(X, y)
        predictions_sgd = log_reg_sgd.predict(X)
        accuracy_sgd = (predictions_sgd == y).mean()
        accuracies_sgd.append((max_iter, accuracy_sgd))

        # Batch Gradient Descent
        log_reg_bgd = LogisticRegressionBatchGD(learning_rate=learning_rate, max_iter=max_iter)
        log_reg_bgd.fit(X, y)
        predictions_bgd = log_reg_bgd.predict(X)
        accuracy_bgd = (predictions_bgd == y).mean()
        accuracies_bgd.append((max_iter, accuracy_bgd))

        # Print accuracy for SGD
        print("Accuracy for Stochastic Gradient Descent:")
        for max_iter, accuracy in accuracies_sgd:
            print(f"Max Iterations: {max_iter}, Accuracy: {accuracy}")

        # Print accuracy for BGD
        print("\nAccuracy for Batch Gradient Descent:")
    # Plotting
    plt.figure(figsize=(10, 6))
    max_iters, accs_sgd = zip(*accuracies_sgd)
    _, accs_bgd = zip(*accuracies_bgd)
    plt.plot(max_iters, accs_sgd, marker='o', label='SGD')
    plt.plot(max_iters, accs_bgd, marker='o', label='BGD')
    plt.xlabel("Max Iterations")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs. Max Iterations for Learning Rate = {learning_rate}")
    plt.legend()
    plt.grid(True)
    plt.show()


# Read the data set
credit_card_data = pd.read_csv("D:/AI Assignment/Credit_card.csv")
pd.set_option('display.max_columns', None)  # To display all columns
pd.set_option('display.max_rows', None)  # To display all rows

# label dataset
label_data = pd.read_csv("D:/AI Assignment/Credit_card_label.csv")

# Merge the dataset
result_df = pd.merge(credit_card_data, label_data, on='Ind_ID')
# Drop irrelevant columns
result_df = result_df.drop(columns=['Ind_ID', 'EMAIL_ID', 'Type_Occupation'])
# Preprocessing
# Handle missing values if any
result_df.dropna(inplace=True)  # Drop rows with missing values

# Encode categorical variables using one-hot encoding
result_df = pd.get_dummies(result_df)

# Ensure that 'Ind_ID', 'EMAIL_ID', 'Type_Occupation' columns are present before dropping them
columns_to_drop = ['Ind_ID', 'EMAIL_ID', 'Type_Occupation']
columns_to_drop = [col for col in columns_to_drop if col in result_df.columns]

# Drop columns if they exist
result_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Split data into features and labels
X = result_df.drop(columns=['label'])  # Features
y = result_df['label']  # Target variable

# Plot accuracy graph for a given learning rate
plot_accuracy(X, y, learning_rate=0.1)


# Read the data set
credit_card_data = pd.read_csv("D:/AI Assignment/Credit_card.csv")
pd.set_option('display.max_columns', None)  # To display all columns
pd.set_option('display.max_rows', None)  # To display all rows

# label dataset
label_data = pd.read_csv("D:/AI Assignment/Credit_card_label.csv")

# Merge the dataset
result_df = pd.merge(credit_card_data, label_data, on='Ind_ID')
# Drop irrelevant columns
result_df = result_df.drop(columns=['Ind_ID', 'EMAIL_ID', 'Type_Occupation'])
# Preprocessing
# Handle missing values if any
result_df.dropna(inplace=True)  # Drop rows with missing values

# Encode categorical variables using one-hot encoding
result_df = pd.get_dummies(result_df)

# Ensure that 'Ind_ID', 'EMAIL_ID', 'Type_Occupation' columns are present before dropping them
columns_to_drop = ['Ind_ID', 'EMAIL_ID', 'Type_Occupation']
columns_to_drop = [col for col in columns_to_drop if col in result_df.columns]

# Drop columns if they exist
result_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Split data into features and labels
X = result_df.drop(columns=['label'])  # Features
y = result_df['label']  # Target variable

# Plot accuracy graph for a given learning rate
plot_accuracy(X, y, learning_rate=0.1)

