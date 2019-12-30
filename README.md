## Inferring-Emotions-in-Audio-Sequences

Emotion detection in audio files using an ensemble of neural networks. I have been familiar with the concept of ensembling neural networks but I wanted to see it's utility in the context of this problem.

I have uploaded the .joblib files for directly using the X (observations) and y (labels) as inferred from the raw data input from Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) dataset. The credit for it goes to *https://github.com/marcogdepinto/Emotion-Classification-Ravdess*

Firstly, I used a XGBoost classifier model, which with some hypertuning tuning gave me an accuracy ~ 86%. 
