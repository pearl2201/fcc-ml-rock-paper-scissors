# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


def p(x):
    if x == "R":
        return 0
    elif x == "S":
        return 1
    elif x == "P":
        return 2
    else:
        raise Exception("Could not convert signal to value " + str(x))


def ip(x):
    if x == 0:
        return "R"
    elif x == 1:
        return "S"
    elif x == 2:
        return "P"
    else:
        raise Exception("Could not convert invert signal to value " + str(x))


def guess_from_ip(x):
    if x == "R":
        return "P"
    elif x == "P":
        return "S"
    elif x == "S":
        return "R"
    else:
        raise Exception("Could not guess value " + str(x))


#model = tf.keras.models.Sequential()
prediction_days = 4
clf = OneVsRestClassifier(SVC())


def define_model3():
    global clf
    del clf
    clf = OneVsRestClassifier(SVC())
    #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=0)
    #clf = OneVsRestClassifier(RandomForestClassifier(random_state=1))


def predict3(opponent_history, my_history):
    global clf
    y = [p(x) for x in opponent_history]
    y2 = [p(x) for x in my_history]
    xT = []
    yT = []
    for idx in range(prediction_days+1, len(y)-1):
        a1 = y2[idx-prediction_days-1:idx-1]
        a2 = y[idx-prediction_days-1:idx-1]
        xT.append(np.concatenate([a1, a2]))
        yT.append(y[idx])
    xT = np.array(xT)
    yT = np.array(yT)
    clf = clf.fit(xT, yT)

    x_test = [np.concatenate([y2[len(opponent_history) - prediction_days -
                                 1:len(opponent_history) - 1],
                              y[len(opponent_history) - prediction_days -
                                1: len(opponent_history) - 1]])]
    x_test = np.array(x_test)
    ind = clf.predict(x_test)

    predict_next_action = ip(ind[0])
    guess = guess_from_ip(predict_next_action)
    #print("predict: {0}, guess: {1}".format(predict_next_action, guess))
    return guess


# def define_model2():
#     global model
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.Input(shape=(prediction_days, 1,)))
#     model.add(tf.keras.layers.LSTM(
#         16, activation='relu', return_sequences=True))
#     model.add(tf.keras.layers.Dropout(0.2))
#     model.add(tf.keras.layers.LSTM(
#         units=32, activation='relu', return_sequences=True))
#     model.add(tf.keras.layers.Dropout(0.2))
#     model.add(tf.keras.layers.LSTM(4, return_sequences=False))
#     model.add(tf.keras.layers.Dropout(0.2))
#     model.add(tf.keras.layers.Dense(32, activation='relu'))
#     model.add(tf.keras.layers.Dropout(0.2))
#     model.add(tf.keras.layers.Dense(3, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam')
#     model.summary()


# def predict2(opponent_history):
#     y = [p(x) for x in opponent_history]
#     y2 = tf.keras.utils.to_categorical(y, num_classes=3)
#     xT = []
#     yT = []
#     for idx in range(prediction_days+1, len(y)-1):
#         xT.append(y[idx-prediction_days-1:idx-1])
#         yT.append(y2[idx])
#     # print(np.shape(xT))
#     # print(np.shape(xT)[1])
#     xT = np.array(xT)
#     yT = np.array(yT)
#     xT = np.reshape(xT, (xT.shape[0], xT.shape[1], 1))
#     # print(np.shape(xT))
#     # print(np.shape(xT)[1])
#     model.fit(xT, yT, epochs=4, batch_size=4, verbose=0)
#     x_test = y[len(opponent_history) - prediction_days -
#                1: len(opponent_history) - 1]
#     x_test = np.array(x_test)
#     x_test = np.reshape(x_test, (1, xT.shape[1], 1))
#     yhat = model.predict(x_test)
#     ind = np.argmax(yhat, axis=1)[0]

#     predict_next_action = ip(ind)
#     guess = guess_from_ip(predict_next_action)
#     print("predict: {0}, guess: {1}".format(predict_next_action, guess))
#     return guess

# def define_model():
#     global model
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.Input(shape=(1,)))
#     model.add(tf.keras.layers.Dense(4, activation='relu'))
#     model.add(tf.keras.layers.Dense(8, activation='relu'))
#     model.add(tf.keras.layers.Dense(3,activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.summary()


# def predict(opponent_history):
#     x = np.array([[i] for i in range(len(opponent_history))])
#     #xT = np.array([[i-2,i-1,i] for i in range(2,len(opponent_history))])
#     y = [p(x) for x in opponent_history]
#     y = tf.keras.utils.to_categorical(y, num_classes=3)
#     model.fit(x, y,epochs=4,batch_size=4,verbose=0, validation_split=0.1)
#     yhat = model.predict([len(opponent_history) + 1])
#     ind = np.argmax(yhat,axis=1)
#     predict_next_action = ip(ind[0])
#     guess = guess_from_ip(predict_next_action)
#     print ("predict: {0}, guess: {1}".format(predict_next_action, guess))
#     return guess


def player(prev_play, opponent_history=[], my_history=[]):
    if len(opponent_history) == 0:
        define_model3()
    if prev_play != "":
        opponent_history.append(prev_play)

    guess = "R"
    if len(opponent_history) > prediction_days + 3:
        #guess = opponent_history[-2]
        guess = predict3(opponent_history, my_history)
    my_history.append(guess)
    return guess
