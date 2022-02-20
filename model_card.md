# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- It is a Logistic regression model, trained with default hyperparamters in scikit-learn 1.0.2  
- Last training 2022-02-17
- Model Version 1.0.0
- Created by Kay Auerbach

## Intended Use
This model should be used to predict if a salary is above 50K per year based off 8 categorical (e.g. workclass, education, marital-status) and 6 continuous attributes (e.g. capital-gain, capital-loss, hours-per-week). 
## Data
The data is published on the UCI Machine Learning Repository https://archive.ics.uci.edu/ml/datasets/census+income.
The target attribute salary is mapped on 1 if it has in the original dataset the Value '>50K' and on 0 in the other case ('<=50K').
The original data set has 48842 rows and 14 attributes and a 80-20 split without stratification was used to create a train and test set.
A One Hot Encoder for the categorical features and a Label Binarizer for the labels was used.

## Metrics
The model was evaluated using the following scores:
- precision:  0.721
- recall:  0.266
- fbeta:  0.389

## Ethical Considerations
Care should be taken to ensure that the model does not discriminate against anyone, especially with regard to gender and origin.
## Caveats and Recommendations
- No hyper parameter tuning and no feature engineering were made
- Try other methods and train other models for comparison
