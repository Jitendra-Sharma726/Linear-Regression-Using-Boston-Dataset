import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --------- 1. Load Dataset ---------
def load_dataset(file_path: str) -> pd.DataFrame:
    """Load CSV dataset into a DataFrame"""
    df = pd.read_csv(file_path)
    return df


# --------- 2. Split Features and Target ---------
def split_features_target(df: pd.DataFrame):
    X = df.drop("medv", axis=1).values
    Y = df["medv"].values
    return X, Y


# --------- 3. Train Test Split ---------
def split_train_test(X, Y, test_size=0.2, random_state=42):
    """Split dataset into training and testing sets"""
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)


# --------- 4. Train Linear Regression Model ---------
def train_model(X_train, Y_train):
    """Train Linear Regression model"""
    # Ensure X_train is 2D
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model



# --------- 5. Predict ---------
def predict(model, X_test):
    """Predict target values for X_test"""
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    return model.predict(X_test)


# --------- 6. Evaluate Model ---------
def evaluate_model(Y_test, Y_pred):
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

    return mse, r2


# --------- 7. Display Results ---------
def display_results(Y_test, Y_pred, num_rows=10):
    """Display actual vs predicted values"""
    results = pd.DataFrame({
        "Actual Price": Y_test,
        "Predicted Price": Y_pred
    })
    print(results.head(num_rows))
    return results

# --------- 8. Plot Regression ---------
def plot_regression(Y_test, Y_pred, save_path: str):
    """Scatter plot of actual vs predicted prices"""
    plt.figure()
    plt.scatter(Y_test, Y_pred, color="blue", label="Predicted vs Actual")
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()],
             color="red", linestyle="--", label="Perfect Fit")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Housing Prices")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


# --------- Main Execution ---------
if __name__ == "__main__":
    dataset_path = os.path.join(BASE_DIR, "dataset.csv")
    plot_path = os.path.join(BASE_DIR, "plot.png")

    # Load dataset
    df = load_dataset(dataset_path)

    # Split into features and target
    X, Y = split_features_target(df)

    # Train test split
    X_train, X_test, Y_train, Y_test = split_train_test(X, Y)

    # Train model
    model = train_model(X_train, Y_train)

    # Predict
    Y_pred = predict(model, X_test)

     # Evaluate
    mse, r2 = evaluate_model(Y_test, Y_pred)

    # Display results
    display_results(Y_test, Y_pred)

    # Plot
    plot_regression(Y_test, Y_pred, plot_path)
    print(f"Plot saved to: {plot_path}")


























