# Sleep Disorder Classification

A COGS 109 course project at UC San Diego. It predicts sleep disorder status from lifestyle and health features using K-nearest neighbors and logistic regression.

## Data

The Sleep Health and Lifestyle dataset (included as a CSV). Features include gender, occupation, BMI category and blood pressure, along with other lifestyle and health measures.

## Method

1. Split blood pressure into systolic and diastolic values.
2. One-hot encode gender, BMI category and occupation, then standardize all features.
3. Evaluate with 5-fold stratified cross-validation.
4. KNN: sweep k from 1 to 49 and compare training accuracy against cross-validated accuracy to show the bias-variance tradeoff.
5. Logistic regression (L2): sweep the regularization strength C from 0.001 to 100.
6. Rank features by the average absolute logistic regression coefficient across classes.

## Output figures

- knn_bias_variance.png: KNN training vs cross-validated accuracy
- lr_cv_accuracy.png: logistic regression cross-validated accuracy vs C
- feature_importance.png: top 15 features

## Run

pip install pandas numpy matplotlib seaborn scikit-learn
python analysis.py
