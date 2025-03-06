# Heart Disease Prediction using Machine Learning

## Overview

This project focuses on predicting the presence of heart disease in patients using various machine learning algorithms. The dataset used in this project is the `heart.csv` dataset, which contains various patient attributes such as age, sex, chest pain type, resting blood pressure, cholesterol levels, and more. The target variable indicates the presence (1) or absence (0) of heart disease.

The project involves the following steps:
1. **Data Loading and Preprocessing**: The dataset is loaded and split into training and testing sets.
2. **Model Training**: Several machine learning models are trained, including:
   - Random Forest Classifier
   - Gaussian Naive Bayes
   - Gradient Boosting Classifier
   - K-Nearest Neighbors (KNN)
   - Logistic Regression
   - Support Vector Classifier (SVC)
3. **Model Evaluation**: The models are evaluated based on their accuracy and recall scores.
4. **Hyperparameter Tuning**: Grid Search Cross-Validation is used to find the best hyperparameters for the Random Forest Classifier.
5. **Feature Importance**: The importance of each feature in the dataset is visualized.
6. **ROC Curve Analysis**: The Receiver Operating Characteristic (ROC) curve is plotted to evaluate the performance of the models.

## Results

The best-performing model after hyperparameter tuning is the Random Forest Classifier, which achieves an accuracy of **98.54%** and a recall score of **98.59%** on the test set. The ROC curve for this model shows an AUC (Area Under the Curve) of **0.99**.

## Feature Importance

The feature importance analysis reveals that the most significant features contributing to the prediction of heart disease are:
1. **cp**  (chest pain type)
2. **thalach** (maximum heart rate achieved)
3. **ca** (number of major vessels colored by fluoroscopy)

## Dependencies

To run this project, you will need the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install these dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Run the Jupyter Notebook or Python script:

   ```bash
   jupyter notebook heart_disease_prediction.ipynb
   ```

   or

   ```bash
   python heart_disease_prediction.py
   ```

3. Follow the instructions in the notebook/script to load the dataset, train the models, and evaluate their performance.

## Dataset

The dataset used in this project is `heart.csv`, which is included in the repository. It contains the following columns:

- `age`: Age of the patient
- `sex`: Sex of the patient (1 = male; 0 = female)
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure (in mm Hg)
- `chol`: Serum cholesterol in mg/dl
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- `restecg`: Resting electrocardiographic results (0-2)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina (1 = yes; 0 = no)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: The slope of the peak exercise ST segment
- `ca`: Number of major vessels (0-3) colored by fluoroscopy
- `thal`: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
- `target`: Presence of heart disease (1 = yes; 0 = no)
