# Diabetes Prediction Using Machine Learning

## Project Overview

This project explores the application of machine learning to predict diabetes risk using health and lifestyle indicators from the CDC's Behavioral Risk Factor Surveillance System.

The goal is to develop and compare multiple classification models to identify patterns associated with diabetes and evaluate their predictive performance.

The project demonstrates practical experience in data preprocessing, supervised machine learning, model evaluation, and Python-based data visualization.

## Personal Motivation

Diabetes has personally affected my family, as both of my grandparents have dealt with the condition. Seeing its impact on their lives sparked my interest in understanding how data and technology could help identify health risks.

As I continued developing my skills in data analytics and machine learning, I wanted to apply what I was learning to a problem that held personal significance.

This project became an opportunity to explore how machine learning algorithms can identify patterns in health and lifestyle data associated with diabetes. By comparing different classification models, I hope to better understand the potential of predictive analytics in healthcare and how data-driven approaches could contribute to earlier risk identification.

Beyond building technical skills, this project represents my interest in using data science to address real-world challenges that affect individuals and families.

## Dataset

**Source:** [CDC Diabetes Health Indicators Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

The project uses the balanced `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` dataset, containing 70,692 observations and 21 health-related predictors.

The target variable, `Diabetes_binary`, represents whether an individual has diabetes (1) or does not have diabetes (0), according to the dataset's labeling.

### Health Indicators

| Feature | Description |
|---|---|
| `HighBP` | Whether the individual has high blood pressure |
| `HighChol` | Whether the individual has high cholesterol |
| `CholCheck` | Whether cholesterol was checked within the past five years |
| `BMI` | Body Mass Index |
| `Smoker` | Whether the individual has smoked at least 100 cigarettes in their lifetime |
| `Stroke` | History of stroke |
| `HeartDiseaseorAttack` | History of coronary heart disease or myocardial infarction |
| `PhysActivity` | Physical activity within the past 30 days, excluding work |
| `Fruits` | Whether the individual consumes fruit at least once daily |
| `Veggies` | Whether the individual consumes vegetables at least once daily |
| `HvyAlcoholConsump` | Heavy alcohol consumption indicator |
| `AnyHealthcare` | Whether the individual has healthcare coverage |
| `NoDocbcCost` | Whether cost prevented a doctor visit in the past 12 months |
| `GenHlth` | Self-reported general health, rated from 1 (excellent) to 5 (poor) |
| `MentHlth` | Number of days of poor mental health in the past 30 days |
| `PhysHlth` | Number of days of poor physical health in the past 30 days |
| `DiffWalk` | Serious difficulty walking or climbing stairs |
| `Sex` | Sex indicator, coded 0 for female and 1 for male |
| `Age` | Age category, coded from 1 to 13 |
| `Education` | Education level, coded from 1 to 6 |
| `Income` | Income category, coded from 1 to 8 |

## Machine Learning Models

Three supervised classification algorithms were implemented and evaluated.

### Logistic Regression

- Establishes a baseline for binary classification.
- Models the relationship between health indicators and diabetes outcomes.

### Random Forest

- Uses an ensemble of decision trees to capture complex relationships.
- Accounts for nonlinear patterns and interactions between predictors.
- Random Forest PKL file is excluded due to size but is generated when training pipeline runs

### Support Vector Machine (SVM)

- Classifies observations by identifying a decision boundary between classes.
- Provides an alternative approach for evaluating classification performance.

## Model Evaluation

Models are evaluated using standard classification metrics:

- **Accuracy:** Percentage of correctly classified observations.
- **Confusion Matrix:** Visualization of true positives, true negatives, false positives, and false negatives.

Confusion matrix visualizations are automatically generated and saved in the `visualizations/` directory.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/JohnnyChengGitHub/diabetes_predictor.git
cd diabetes_predictor
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment.

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required Python libraries:

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the Dataset

Download the CDC Diabetes Health Indicators dataset from the UCI Machine Learning Repository.

Place the following file inside the `data/` directory:

```text
diabetes_binary_5050split_health_indicators_BRFSS2015.csv
```

## Usage

### 1. Run the Machine Learning Pipeline

From the project's root directory, execute:

```bash
python -m src.main
```

The project runs the implemented classification workflow using the CDC diabetes dataset.

### 2. Review Model Performance

Review the test accuracy results printed in the terminal for Logistic Regression, Random Forest, and Support Vector Machine (SVM).

### 3. View Confusion Matrices

Confusion matrices are automatically generated and saved in the `visualizations/` directory for each model.

### 3. View Confusion Matrices

Generated confusion matrices are saved in the `visualizations/` directory.

These visualizations provide a breakdown of each model's classification predictions and errors. The matrices are automatically generated and saved in the visualizations directory for each model.


## Project Structure

```text
diabetes_predictor/
├── data/                  # CDC diabetes dataset
├── models/                # Saved machine learning models
├── src/
│   ├── data_processing.py # Dataset loading and preprocessing
│   ├── main.py            # Main execution script
│   ├── models.py          # Model training and evaluation
│   └── visualize.py       # Confusion matrix visualization
├── visualizations/        # Generated confusion matrices
├── .gitignore
├── requirements.txt
└── README.md
```


## Disclaimer
This project is intended for educational and analytical purposes only. The models are not designed or validated for clinical diagnosis or medical decision-making.
