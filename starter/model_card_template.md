# Model Card

GradientBoostingClassifier trained on the [US census dataset](https://archive.ics.uci.edu/ml/datasets/census+income).

## Model Details

- The model: `GradientBoostingClassifier`
- Train: `80% training data`
- Test: `20% for testing data`

## Intended Use

The model aims to predict whether a person earns more than $50k based on census information.

## Training Data

The data: [US census dataset](https://archive.ics.uci.edu/ml/datasets/census+income).
The data was splited: 80% training and 20% testing
Categorical features :

- `workclass`
- `education`
- `marital_status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `native_country`

## Evaluation Data

I used 20% of the data to evaluate the model

## Metrics

Using all data:

- train:

* `precision`: 79%
* `recall`: 61%
* `fbeta`: 69%

- test:

* `precision`: 79%
* `recall`: 59%
* `fbeta`: 67%

## Ethical Considerations

The Dataset contains data related race, gender and origin country. This may drive to a model that discriminate people and because of that, further investigation should be done before use it.

## Caveats and Recommendations

The model could be improved by analysing feature importance and performing feature engineering on the dataset.
