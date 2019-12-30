# Inferring-Emotions-in-Audio-Sequences

Emotion detection in audio files using an ensemble of neural networks. I have been familiar with the concept of ensembling neural networks but I wanted to see it's utility in the context of this problem.

I have uploaded the .joblib files for directly using the X (observations) and y (labels) as inferred from the raw data input from Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) dataset. The credit for it goes to *https://github.com/marcogdepinto/Emotion-Classification-Ravdess* (It's readme details the finer aspects of the dataset with it's characterstics)

## Tools and languages (Recommended):
* Python 3.7
* Google Colab
* Google Drive
* Jupyter Notebook
* Tensorflow- Keras
* Numpy

## Addressing Class Imbalance:
![alt text](https://user-images.githubusercontent.com/45323656/71593192-14ac0180-2b01-11ea-9321-8f7f2be6c1ca.png)

Keeping in mind that the dataset is imbalanced w.r.t to the labels available. SMOTE is used in here to balance it. Mind here, the SMOTE refers to Synthetic Minority Oversampling Technique. Thus, it doesn't reduce the majority samples.

![alt text](https://user-images.githubusercontent.com/45323656/71593279-6f455d80-2b01-11ea-9085-47804835f89c.png)

## XGBClassifier: 
Firstly, I used a XGBoost classifier model, which with some hypertuning tuning gave me an accuracy ~ 89%. The classifer hyperparamters are detailed below: 
``` python
params = {'max_depth': 7,
          'gamma': 0.01,
          'n_estimators': 200,
          'learning_rate':0.01,
          'subsample': 0.8,
          'eta': 0.1,
          'eval_metric': 'rmse'}

classifier = XGBClassifier(**params).fit(X_train, y_train)
```
Following is the classification report generated w.r.t the XGB classifier model's performance.
```
              precision    recall  f1-score   support

           0       0.97      0.94      0.96       161
           1       0.94      0.93      0.93       148
           2       0.79      0.96      0.87       125
           3       0.77      0.93      0.84       153
           4       0.92      0.97      0.94       146
           5       0.80      0.91      0.85       125
           6       1.00      0.75      0.86       194
           7       0.97      0.82      0.89       182

    accuracy                           0.89      1234
   macro avg       0.90      0.90      0.89      1234
weighted avg       0.91      0.89      0.89      1234

By the confusion matrix, the accuracy of the XGB model is = 0.8922204213938412
```
## Single Deep Neural Network:
I used an 18-layer neural network which could reach an accuracy of ~92%. It's model architecture and performance details are given below:
