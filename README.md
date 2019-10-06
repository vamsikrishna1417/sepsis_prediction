# sepsis_prediction
Implemented five different machine learning algorithms based on the data collected from physionet sepsis prediction challenge to identify a patient's risk of sepsis and make a prediction of sepsis before 4 and 6 hours.  

Used various techniques which includes :

Oversampling and Undersampling - To preprocess imbalanced data and to overcome the sensitivity of classifiers,

Interpolation and backfilling - To fill the missing data,

Bucketized column feature engineering followed by OneHotEncoding - To represent categorical data, Feature selection - To reduce total number of features,

Standard scaling - To normalize the data,

Bootstrap aggregating (Bagging) - To Enhance the stability and accuracy of the models.

Ultimately, able to predict the onset of sepsis with an accuracy of 78-80%.
