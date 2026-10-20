
from src.data_processing import load_data, preprocess_data
import src.models as model
from src.visualize import plot_confusion_matrix


def main():
    # Load and preprocess the diabetes dataset
    data = load_data()
    x_train_scaled, x_test_scaled, y_train, y_test = preprocess_data(data)

    # Train and save Logistic Regression
    model_lr = model.train_logistic_regression(x_train_scaled, y_train)
    model.save_model(model_lr, "logistic_regression_model")

    # Train and save Random Forest
    model_rf = model.train_random_forest(x_train_scaled, y_train)
    model.save_model(model_rf, "random_forest_model")

    # Train and save SVM
    model_svm = model.train_svm(x_train_scaled, y_train)
    model.save_model(model_svm, "svm_model")

    # Load and evaluate each trained model
    for model_name in [
        "logistic_regression_model",
        "random_forest_model",
        "svm_model"
    ]:
        loaded_model = model.load_model(model_name)

        accuracy, matrix = model.evaluate_model(
            loaded_model, x_test_scaled, y_test
        )

        print(f"{model_name} Test Accuracy: {accuracy * 100:.2f}%")

        plot_confusion_matrix(matrix, model_name)


if __name__ == "__main__":
    main()
