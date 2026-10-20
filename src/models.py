from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,accuracy_score
import joblib
import os


def train_logistic_regression(x_train_scaled, y_train):
    model_lr = LogisticRegression()
    model_lr.fit(x_train_scaled, y_train)
    return model_lr

def train_random_forest(x_train_scaled, y_train):
    from sklearn.ensemble import RandomForestClassifier
    model_rf = RandomForestClassifier()
    model_rf.fit(x_train_scaled, y_train)
    return model_rf

def train_svm(x_train_scaled, y_train):
    from sklearn.svm import SVC
    model_svm = SVC(probability=True)
    model_svm.fit(x_train_scaled, y_train)
    return model_svm

#evaluate model performance
def evaluate_model(model, x_test_scaled, y_test):
    #make predictions using unseen test data
    y_test_predictions = model.predict(x_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_predictions)
    matrix = confusion_matrix(y_test, y_test_predictions)

    return test_accuracy, matrix

#save trained model
def save_model(model, model_name):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{model_name}.pkl")
    print(f"Model saved as {model_name}")

#load trained model
def load_model(model_name):
    model = joblib.load(f"models/{model_name}.pkl")
    print(f"Model loaded from {model_name}")
    return model