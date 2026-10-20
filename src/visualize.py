import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for environments without a display
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(matrix, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'visualizations/confusion_matrix_{model_name}.png')  # Save the plot as a PNG file
    plt.close()  # Close the plot to free up memory