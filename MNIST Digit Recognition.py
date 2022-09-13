from tensorflow import keras
import numpy
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def load_mnist_data() -> tuple:
    (trainX, trainY), (testX, testY) = keras.datasets.mnist.load_data()
    
    trainX = trainX.reshape(60000, 28, 28, 1)
    testX = testX.reshape(10000, 28, 28, 1)

    trainX = trainX.astype("float")
    testX = testX.astype("float")

    trainX = trainX / 255.0
    testX = testX / 255.0

    trainY = keras.utils.to_categorical(trainY)
    testY = keras.utils.to_categorical(testY)

    return trainX, trainY, testX, testY


def make_model():

    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=['accuracy'])

    return model


def test_model(dataX: numpy.ndarray, dataY: numpy.ndarray, folds: int, epoch: int, batch: int) -> None: 

    accuracies, histories = list(), list()
    kfold = KFold(folds, shuffle=True, random_state=1)


    count = 1
    for train_ix, test_ix in kfold.split(dataX):

        print("Training model with fold number %d" % count)

        model = make_model()

        trainX = dataX[train_ix]
        trainY = dataY[train_ix]
        testX = dataX[test_ix]
        testY = dataY[test_ix]

        history = model.fit(trainX, trainY, epochs=epoch, batch_size=batch, validation_data=(testX, testY), verbose=1)

        loss, accuracy = model.evaluate(testX, testY, verbose=1)
        accuracies.append(accuracy)
        histories.append(history)

        count += 1


    print("Model trained!")

    for x in range(len(histories)):

        plt.subplot(2,1,1)
        plt.title("Cross Entropy Loss")
        plt.plot(histories[x].history['loss'], color="orange", label="Training Data")
        plt.plot(histories[x].history["val_loss"], color="cyan", label="Testing data")


        plt.subplot(2,1,2)
        plt.title("Classification Accuracy")
        plt.plot(histories[x].history['accuracy'], color="orange", label="Training Data")
        plt.plot(histories[x].history["val_accuracy"], color="cyan", label="Testing data")

    
    plt.show()

    plt.boxplot(accuracies)
    plt.title("Model Accuracies")
    plt.show()

    print("Mean: %.3f" % numpy.mean(accuracies))
    print("Standard deviation: %.3f" % numpy.std(accuracies))
    print("Number of model iteratons: %d" % len(accuracies))




def testing_harness() -> None:
    trainX, trainY, testX, testY = load_mnist_data()
    
    inp_fold = 0

    while (True):
        try:
            inp_fold = int(input("Enter the number of folds for k-fold cross validation: "))
        except:
            print("Invalid input! Try again.")
        else:
            if (inp_fold > 0):
                break
            else:
                print("Fold number cannot be less than 1!")

    epoch = 0

    while (True):
        try:
            epoch = int(input("Enter the number of epochs to train the data for: "))
        except:
            print("Invalid input! Try again.")
        else:
            if (epoch > 0):
                break
            else:
                print("Epoch number cannot be less than 1!")

    
    batch = 0

    while (True):
        try:
            batch = int(input("Enter the batch size: "))
        except:
            print("Invalid input! Try again.")
        else:
            if (batch > 0):
                break
            else:
                print("Batch size cannot be less than 1!")

    
    test_model(trainX, trainY, inp_fold, epoch, batch)



def output_model() -> None:

    epoch = 0

    while (True):
        try:
            epoch = int(input("Enter the number of epochs to train the data for: "))
        except:
            print("Invalid input! Try again.")
        else:
            if (epoch > 0):
                break
            else:
                print("Epoch number cannot be less than 1!")

    
    batch = 0

    while (True):
        try:
            batch = int(input("Enter the batch size: "))
        except:
            print("Invalid input! Try again.")
        else:
            if (batch > 0):
                break
            else:
                print("Batch size cannot be less than 1!")



    trainX, trainY, testX, testY = load_mnist_data()
    model = make_model()

    print("Training model")
    model.fit(trainX, trainY, epochs=epoch, batch_size=batch, verbose=1)

    print("Testing model")
    loss, accuracy = model.evaluate(testX, testY, verbose=1)

    print("Model tested with an accuracy of %.3f and loss of %.3f. Exporting model." % (accuracy, loss))
    model.save("MNIST_model.h5")


def main() -> None:

    y_n = ""

    while (True):
        y_n = str(input("Enter Y for testing harness, or N to train and export final model: "))

        if (y_n in ("Y", "y")):
            testing_harness()
        
        elif (y_n in ("N", "n")):
            output_model()
            break

        else:
            print("Invalid output! Try again")


main()
