import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow


from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)

# mlflow tracking.
import dagshub
dagshub.init(repo_owner='kush2501', repo_name='mlflow-dagshub-demo', mlflow=True)


# mlflow tracking uri.
mlflow.set_tracking_uri("https://dagshub.com/kush2501/mlflow-dagshub-demo.mlflow")


iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 1

# Apply mlflow.
mlflow.set_experiment("Iris-dt")

with mlflow.start_run():

    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    mlflow.log_metric("Accuracy - ", accuracy_score(y_pred, y_test))
    mlflow.log_param("Max Depth - ", max_depth)

    # Create Confusion Matrix.
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    # save the figure.
    plt.savefig("confusion_matrix.png")

    # mlflow code.
    mlflow.log_artifact("confusion_matrix.png")

    # Save code.
    mlflow.log_artifact(__file__)

    # log model.
    mlflow.sklearn.log_model(dt, name="Decision Tree")

    # sets tags.
    mlflow.set_log("Author", "kushx")