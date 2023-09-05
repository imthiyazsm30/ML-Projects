
TIME SERIES DATASET

OBJECTIVE:
The dataset contains past 5 years stock record of google and using this we will be predicting stock price for the next year using RNN-LSTM model.

The RNN-Long Short-term memory model is fed and trained with past data and the objective here is to predict the stock price for the forthcoming days.


CREATING NEW X_TRAIN,Y_TRAIN VARIABLES:
We will be taking the reference of past 60 days of data to predict the future stock price.

It is observed that taking 60 days of past data gives us best results.

The input values(X_TRAIN) must be the stock prices at time t(0-59) and the output values should be the stock prices at time t+1(Y_TRAIN)


MODEL:

1)Import the Keras packages(Tensorflow-keras)

2)Initalize RNN(Sequential)

3)Add the first LSTM Layer and Dropout regularization

4)Add the second layer of LSTM and dropout regularization

5)Add the third layer of LSTM and dropout regularization

6)Add the output layer and compile the RNN

7)Fit the model to the training set

8)Make predictions and visualize the output

