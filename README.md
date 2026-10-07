```markdown
# Adult Census ML Pipeline

A Machine Learning project for predicting whether a person's annual income is `<=50K` or `>50K` using the Adult Census Income dataset.

## Project Overview

This project builds, evaluates, tunes, and tracks Machine Learning classification pipelines for the Adult Census Income dataset.

The project covers the complete Machine Learning workflow:

- Data preprocessing
- Feature transformation
- Logistic Regression
- Random Forest Classifier
- Model evaluation
- Cross-validation
- Hyperparameter tuning using GridSearchCV
- MLflow experiment tracking
- Model saving and loading
- Git and GitHub version control

## Dataset

The project uses the Adult Census Income dataset.

The target variable contains two classes:

- `<=50K` — Income is less than or equal to 50K
- `>50K` — Income is greater than 50K

The dataset contains demographic and employment-related features, including:

- Age
- Workclass
- Final Weight
- Education
- Education Number
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country

## Data Preprocessing

The preprocessing pipeline handles numerical and categorical features automatically.

### Numerical Features

The numerical features are processed using:

- Missing value imputation
- StandardScaler

### Categorical Features

The categorical features are processed using:

- Missing value imputation
- OneHotEncoder

The preprocessing steps and machine learning model are combined into a single Scikit-learn Pipeline.

```text
Raw Data
    ↓
Data Preprocessing
    ↓
Numerical / Categorical Transformation
    ↓
Machine Learning Model
    ↓
Prediction
    ↓
Evaluation
```

## Machine Learning Models

Two classification models were trained and evaluated.

### 1. Logistic Regression

Logistic Regression was used as the baseline classification model.

### 2. Random Forest Classifier

Random Forest Classifier was trained and compared with Logistic Regression.

Based on the cross-validation results, Logistic Regression achieved the better F1-score.

## Model Evaluation

The following evaluation metrics were used:

- Accuracy
- Precision
- Recall
- F1-score

The positive class for F1-score evaluation is:

```text
>50K
```

## Cross-Validation Results

| Model | Mean F1 | Std F1 |
|---|---:|---:|
| Logistic Regression | 0.78 | 0.00323 |
| Random Forest | 0.73 | 0.006 |

Logistic Regression achieved the highest mean F1-score with a low standard deviation across the cross-validation folds.

## Test Result

The selected Logistic Regression pipeline achieved:

```text
Test F1 = 0.6581
```

## Hyperparameter Tuning

GridSearchCV was used to tune the Logistic Regression model.

The following hyperparameters were explored:

```python
param_grid = {
    "classifier__C": [0.1, 1, 10],
    "classifier__solver": ["liblinear", "lbfgs"]
}
```

The GridSearchCV experiments were also tracked using nested MLflow runs.

## MLflow Experiment Tracking

MLflow was used to track and compare the machine learning experiments.

The experiment name is:

```text
Adult_Census_Pipeline
```

A local SQLite database is used as the MLflow backend:

```text
sqlite:///mlruns.db
```

The experiments track:

- Model parameters
- Accuracy
- Precision
- Recall
- F1-score
- Cross-validation results
- Model artifacts
- Experiment tags

MLflow was also used to compare different models and hyperparameter combinations.

## Model Saving

The complete preprocessing and classification pipeline is saved as one model.

This allows new raw data to be passed directly to the saved pipeline without manually applying preprocessing steps.

Example:

```python
import joblib

joblib.dump(
    pipeline,
    "adult_income_pipeline.joblib"
)
```

The saved pipeline can later be loaded:

```python
loaded_pipeline = joblib.load(
    "adult_income_pipeline.joblib"
)
```

## Example Prediction

The trained pipeline can receive raw input data directly.

Example:

```python
new_data = pd.DataFrame({
    "age": [25, 45],
    "workclass": ["Private", "Private"],
    "fnlwgt": [226802, 150000],
    "education": ["11th", "Bachelors"],
    "education-num": [7, 13],
    "marital-status": ["Never-married", "Married-civ-spouse"],
    "occupation": ["Machine-op-inspct", "Exec-managerial"],
    "relationship": ["Own-child", "Husband"],
    "race": ["Black", "White"],
    "sex": ["Male", "Male"],
    "capital-gain": [0, 0],
    "capital-loss": [0, 0],
    "hours-per-week": [40, 50],
    "native-country": ["United-States", "United-States"]
})

predictions = loaded_pipeline.predict(new_data)

print(predictions)
```

Possible output:

```text
['<=50K' '>50K']
```

## Project Workflow

The project follows these main stages:

```text
1. Load Adult Census Dataset
        ↓
2. Data Preprocessing
        ↓
3. Build Baseline Pipeline
        ↓
4. Train Logistic Regression
        ↓
5. Train Random Forest
        ↓
6. Evaluate Models
        ↓
7. Cross-Validation
        ↓
8. GridSearchCV Hyperparameter Tuning
        ↓
9. Select Best Pipeline
        ↓
10. Track Experiments with MLflow
        ↓
11. Save Final Pipeline
        ↓
12. Version Control with Git
```

## Git Workflow

Git was used to manage the project development and version history.

The repository includes:

- Initial baseline implementation
- GridSearchCV tuning
- Experimental branch
- Merge into the main branch
- Final pipeline
- Version tag `v1.0`

The tuning branch was created using:

```bash
git checkout -b experiment/gridsearch-tuning
```

After completing the tuning experiments, the branch was merged into `main`.

The final version was tagged:

```bash
git tag -a v1.0 -m "Final Adult Census ML pipeline"
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- Joblib
- Jupyter Notebook
- Git
- GitHub

## Project Structure

```text
adult-census-ml-pipeline/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── notebooks/
│   └── adult_census_pipeline.ipynb
│
├── models/
│   └── adult_income_pipeline.joblib
│
└── ...
```

MLflow tracking files, SQLite databases, Python cache files, and Jupyter checkpoint files are excluded from Git using `.gitignore`.

## Final Conclusion

After comparing Logistic Regression and Random Forest, Logistic Regression achieved the better cross-validation F1-score.

The final solution uses a complete Scikit-learn pipeline that combines data preprocessing and model training. This makes the model easier to reuse and ensures that new data receives the same preprocessing steps used during training.

MLflow was used to track experiments, parameters, metrics, and model artifacts, while Git and GitHub were used for version control and project management.

## Author

**Omar Elshnofy**

Computer Engineering Student  
Machine Learning & AI

GitHub:  
https://github.com/omarelshnofy

LinkedIn:  
www.linkedin.com/in/omar-elshnofy
```