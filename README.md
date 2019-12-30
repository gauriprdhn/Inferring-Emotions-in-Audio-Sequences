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
I used an 18-layer neural network which could reach an accuracy of ~92% with learning rate of 5e-5 and epochs = 100. It's model architecture details are given below:
``` 
------------------MODEL SUMMARY------------------
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_1 (Conv1D)            (None, 40, 128)           768       
_________________________________________________________________
activation_1 (Activation)    (None, 40, 128)           0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 40, 128)           82048     
_________________________________________________________________
activation_2 (Activation)    (None, 40, 128)           0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 40, 128)           0         
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 5, 128)            0         
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 5, 128)            82048     
_________________________________________________________________
activation_3 (Activation)    (None, 5, 128)            0         
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 5, 128)            82048     
_________________________________________________________________
activation_4 (Activation)    (None, 5, 128)            0         
_________________________________________________________________
conv1d_5 (Conv1D)            (None, 5, 128)            82048     
_________________________________________________________________
activation_5 (Activation)    (None, 5, 128)            0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 5, 128)            0         
_________________________________________________________________
conv1d_6 (Conv1D)            (None, 5, 128)            82048     
_________________________________________________________________
flatten_1 (Flatten)          (None, 640)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 5128      
_________________________________________________________________
activation_6 (Activation)    (None, 8)                 0         
=================================================================
Total params: 416,136
Trainable params: 416,136
Non-trainable params: 0
```
We used accuracy/los curves for training and validation sets and classification report was generated to assess model performance.

![alt text](https://user-images.githubusercontent.com/45323656/71594475-11ffdb00-2b06-11ea-87bc-3673a5aeff33.png)

![alt text](https://user-images.githubusercontent.com/45323656/71594482-1cba7000-2b06-11ea-92ab-f8f896468c32.png)

```
               precision    recall  f1-score   support

           0       0.97      0.96      0.97        76
           1       0.92      0.87      0.89        67
           2       0.90      0.88      0.89        73
           3       0.90      0.86      0.88        93
           4       0.97      0.96      0.96        91
           5       0.85      0.86      0.86        66
           6       0.88      1.00      0.94        67
           7       0.98      1.00      0.99        84

    accuracy                           0.92       617
   macro avg       0.92      0.92      0.92       617
weighted avg       0.92      0.92      0.92       61
```
## Deep Ensemble Network:

I tried a simple ensemble using the abovementioned neural network being re-initialised and trained for 5 iterations, with each iteration giving us a new model. Once, we had the ensemble of 5 models, we used ensemble majority voting on the predictions over test set for each model to get the final predictions.
