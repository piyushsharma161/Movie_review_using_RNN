# Movie_review_using_RNN
Predict review type if it is positive or negative

text_classification has all the ML pipeline i.e. exploratory data analysis, data cleaning, model training, saving model etc.

I have used tensorflow recurrent neural network(RNN) with embedding layer to train the model.

Saved the model and word tokenizer, so that they can be used for prediction without retraing the model as traing model may takes hours to complete.

Created deploy.py file which can be used to predict the review type using command prompt. It used saved model.

Use link : https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews#IMDB%20Dataset.csv to dowload the dataset.

Please refer my Github reposiory "movie_review_deploy" to deploy this model locally or globally.
