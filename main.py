import numpy as np
from time import time


class Hyper_Params:
    def __init__(self, epochs, learning_rate, layer_size, activation_function, validation_set_size,
                 input_size, output_size):
        """
        the hyper params class containing all the important information regarding our parameters.
        :param epochs: the amount of epochs we would like to perform.
        :param learning_rate: the learning rate.
        :param layer_size: the hidden layer size.
        :param activation_function: our activation function.
        :param validation_set_size: the % size of the validation set we are going to test against.
        :param input_size: the size of a single input.
        :param output_size: the out put size - amount of classes we can classify example as.
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden_layer_size = layer_size
        self.activation_function = activation_function[0]
        self.function_derivative = activation_function[1]
        self.validation_set_size = validation_set_size
        self.input_size = input_size
        self.output_size = output_size

    def set_epochs_num(self, epochs):
        """
        the function sets the # of epochs.
        :param epochs: new amount of epochs we want to perform.
        """
        self.epochs = epochs

    def set_learning_rate(self, new_rate):
        """
        th sets the learning rate.
        :param new_rate: the new learning rate.
        """
        self.learning_rate = new_rate

    def set_layer_size(self, new_size):
        """
        the function sets the hidden layer size.
        :param new_size: the new layer size we wish to implement.
        """
        self.hidden_layer_size = new_size


def ReLU(x):
    """
    the ReLU mathematical function.
    :param x: the input variable
    :return: output = 0 if x<0 else x.
    """
    return np.maximum(0, x)


def ReLU_Derivative(x):
    """
    the derivative of the function ReLU.
    :param x: the input.
    :return: returns 0 if x<0 else 1
    """
    x[x < 0] = 0
    x[x > 0] = 1
    return x


def softmax(x):
    """
    the softmax function, used in the final layer of our neural network classifier.
    :param x: the x matrix.
    :return: returning a real value for each i in the matrix.
    """
    return np.exp(x) / np.sum(np.exp(x))


def ln(x):
    """
    a simple log function.
    :param x: input variable.
    :return: output y.
    """
    n = 1000.0
    return n * ((x ** (1 / n)) - 1)


def load_data_from_file():
    """
    the function loads data from the Fashion-MNIST database.
    :return: 3 data sets , data set of images, data set of classification,
     and a data set we are going to test our trained neural network on.
    """
    train_x = np.loadtxt("train_x")
    train_y = np.loadtxt("train_y")
    test_x = np.loadtxt("test_x")
    return train_x, train_y, test_x


class Neural_Network:
    def __init__(self, hyper_parameters):
        """
        the neural network class, containing all the weights biases and functions related to the network.
        :param hyper_parameters: the hyper parameters we are going to use.
        """
        self.hyper_parameters = hyper_parameters

        # initializing weights and bases
        self.b1, self.b2, self.w1, self.w2 = self.init_weights_and_biases()

        self.best_parameters = [self.b1, self.b2, self.w1, self.w2]
        self.current_accuracy = 0
        self.best_accuracy = self.current_accuracy

    def init_weights_and_biases(self):
        """
        the function initializes our starting weights and biases.
        :return: the initialized weights and biases.
        """
        b1 = np.random.uniform(-0.2, 0.2, self.hyper_parameters.hidden_layer_size)
        b2 = np.random.uniform(-0.8, 0.8, self.hyper_parameters.output_size)
        w1 = np.random.uniform(-0.01, 0.01, (self.hyper_parameters.hidden_layer_size,
                                             self.hyper_parameters.input_size))
        w2 = np.random.uniform(-0.2, 0.2, (self.hyper_parameters.output_size,
                                           self.hyper_parameters.hidden_layer_size))
        return b1, b2, w1, w2

    # changing the weights and biases to the best option we found so far.
    def set_best_parameters(self):
        """
        the function sets the best weights and bases that we had so far
        as the current weights and biases, so that we can use them.
        """
        self.b1 = self.best_parameters[0]
        self.b2 = self.best_parameters[1]
        self.w1 = self.best_parameters[2]
        self.w2 = self.best_parameters[3]

    # defining the best achieved accuracy
    def set_best_accuracy(self, new_best):
        """
        the function sets the new acquired best accuracy.
        :param new_best: the new best accuracy.
        """
        self.best_accuracy = new_best

    def set_current_accuracy(self, current_accuracy):
        """
        the function sets the current accuracy.
        :param current_accuracy: the current accuracy.
        """
        self.current_accuracy = current_accuracy

    def forward(self, f):
        """
        the forward function calculating the function result + prediction for specific item from data.
        :param f: the item from data set we want to predict result for.
        :return: the activation function result + the prediction matrix.
        """
        act_func_result = self.hyper_parameters.activation_function(f)
        g = self.w2.dot(act_func_result) + self.b2
        prediction = softmax(g)
        return act_func_result, prediction

    def calculate_loss(self, prediction, y):
        """
        the function calculates the loss of a specific prediction.
        :param prediction: the prediction we want to calculate loss for.
        :param y: the correct classification.
        :return: the negative log of the prediction.
        """
        return -ln(prediction[y])

    def back_propagation(self, x, y):
        """
        the function calculates a gradient that is needed in the calculation
        of the weights to be used in the neural network.
        :param x: the item we want to classify.
        :param y: the correct classification.
        :return: returning the loss and the tuned weights and biases.
        """
        f = self.w1.dot(x) + self.b1
        act_func_result, prediction = self.forward(f)
        loss = self.calculate_loss(prediction, y)

        gw2 = np.outer(prediction, act_func_result)
        gw2[y, :] -= act_func_result

        gb2 = np.copy(prediction)
        gb2[y] -= 1

        layer1 = self.w1.dot(x) + self.b1

        dl_dg = prediction.dot(self.w2) - self.w2[y, :]
        dg_db1 = self.hyper_parameters.function_derivative(layer1)

        gb1 = dl_dg * dg_db1
        gw1 = np.outer(gb1, x)

        return loss, gb1, gb2, gw1, gw2

    def update_weights(self, gb1, gb2, gw1, gw2):
        """
        the function updates the weights and biases of the neural network.
        :param gb1: bias # 1
        :param gb2: bias # 2
        :param gw1: weight # 1
        :param gw2: weight # 2
        """
        learning_rate = self.hyper_parameters.learning_rate
        self.b1 -= learning_rate * gb1
        self.b2 -= learning_rate * gb2
        self.w1 -= learning_rate * gw1
        self.w2 -= learning_rate * gw2

    def train(self, train_x, train_y, validation_x, validation_y):
        """
        the training function, where the training of the neural network happens.
        :param train_x: the training data set.
        :param train_y: the correct classification for the training data set.
        :param validation_x: the data set we are going to test our prediction % on.
        :param validation_y: the correct classification of the validation_x data set.
        """
        print "--epoch---train_loss---dev_loss---accuracy---epoch_time--"
        indices = range(train_x.shape[0])
        for i in xrange(self.hyper_parameters.epochs):
            np.random.shuffle(indices)
            start = time()
            total_loss = 0.0
            a = 0
            # going over the items in the data set.
            while a < len(indices):
                x = train_x[a]
                y = int(train_y[a])
                # using forward & back propagation to learn & tune our weights and biases.
                loss, gb1, gb2, gw1, gw2 = self.back_propagation(x, y)
                total_loss += loss
                # updating the weights and biases after each item.
                self.update_weights(gb1, gb2, gw1, gw2)
                a += 1
            # running an accuracy test of the network against a part of the data set.
            avg_loss, accuracy = self.test_on_validation(validation_x, validation_y)
            self.set_current_accuracy(accuracy)
            # checking if the network surpassed current accuracy.
            if self.current_accuracy >= self.best_accuracy:
                self.best_accuracy = self.current_accuracy
                self.best_parameters = [self.b1, self.b2, self.w1, self.w2]
                self.set_best_parameters()
            print (i, total_loss / len(indices), avg_loss, self.current_accuracy * 100, time() - start)
        print "----------------------------"

    def test_on_validation(self, validation_x, validation_y):
        """
        the function runs an accuracy test of the neural network against the provided data set.
        :param validation_x: the data set of items.
        :param validation_y: the data set of classification for the validation_x data set.
        :return: the amount of correct classification and the total loss.
        """
        total_loss = 0.0
        example_count = 0.0
        correct_prediction = 0.0
        for x, y in zip(validation_x, validation_y):
            y = int(y)
            f = self.w1.dot(x) + self.b1
            act_func_result, prediction = self.forward(f)
            loss = self.calculate_loss(prediction, y)
            total_loss += loss
            example_count += 1
            correct_prediction += prediction.argmax() == y

        sum_loss = total_loss / example_count
        sum_correct = correct_prediction / example_count

        return sum_loss, sum_correct

    def write_prediction_to_file(self, array):
        """
        the function runs against the test_x data set and writes the classification
        that the neural network gives the data set to file.
        :param test_x: the data set we are going over.
        """

        file = open("test.pred", "w")
        for i in range(len(array)):
            file.write(array[i] + "\n")
        file.close()

    def predict_for_set(self, test_x):
        """
        the function predicts the results for provided data set.
        :param test_x: the provided data set.
        :return: a list containing the prediction for each item in the set.
        """
        array_prediction = []
        i = 0
        for x in test_x:
            f = self.w1.dot(x) + self.b1
            act_func_result, prediction = self.forward(f)
            final_prediction = str(prediction.argmax())
            array_prediction.insert(i,final_prediction)
            i += 1
        return array_prediction


def normalize_data_sets(train_x, validation_x, test_x):
    """
    the function normalizes the data set.s
    :param train_x:
    :param validation_x:
    :param test_x:
    :return:
    """
    return train_x / 255.0, validation_x / 255.0, test_x / 255.0


def main():
    # hyper parameters.
    validation_set_size = 0.2
    input_size = 28 * 28
    output_size = 10
    epochs = 40
    learning_rate = 0.01
    layer_size = 200
    activation_function = [ReLU, ReLU_Derivative]

    # creating an object to hold our parameters.
    hyper_parameters = Hyper_Params(epochs, learning_rate, layer_size, activation_function, validation_set_size,
                                    input_size, output_size)

    # load data.
    train_x, train_y, test_x = load_data_from_file()

    # shuffling the data sets before separating into learning set and testing set.
    train_x, train_y = shuffle_data(train_x, train_y)

    # training & validation set creation.
    train_x, train_y, validation_x, validation_y = split_sets(train_x, train_y, hyper_parameters)

    # normalizing the data.
    train_x, validation_x, test_x = normalize_data_sets(train_x, validation_x, test_x)

    # creating the neural network.
    network = Neural_Network(hyper_parameters)

    # training the neural network.
    network.train(train_x, train_y, validation_x, validation_y)

    # updating the current weight vector before predicting to file our final prediction.
    print ("The best accuracy is ", network.best_accuracy * 100)
    network.set_best_parameters()

    # running our module on the test set and returning the prediction.
    array = network.predict_for_set(test_x)

    # writing our prediction to the test.pred file.
    network.write_prediction_to_file(array)


def split_sets(train_x, train_y, hyper_parameters):
    """
    the function splits the data sets.
    :param train_x: the data set containing the items.
    :param train_y: the data set containing the classifications.
    :param hyper_parameters: the hyper parameters we are using.
    :return:
    """
    data_set_len = len(train_x)
    size_of_training_set = data_set_len * float(1.0 - hyper_parameters.validation_set_size)
    size_of_training_set = int(size_of_training_set)
    training_set_x = train_x[:size_of_training_set]
    training_set_y = train_y[:size_of_training_set]
    validation_set_x = train_x[size_of_training_set:]
    validation_set_y = train_y[size_of_training_set:]

    return training_set_x, training_set_y, validation_set_x, validation_set_y


def shuffle_data(train_x, train_y):
    """
    the function shuffles the data together to create an unique combination for training.
    :param train_x: the data set containing the items.
    :param train_y: the data set containing the classifications.
    :return: the shuffled data sets.
    """
    vector_y = train_y.flatten()
    vector_y = vector_y.reshape(-1, 1)
    # concatenate images with their classification.
    data = np.concatenate((train_x, vector_y), axis=1)
    # shuffling the data set.
    np.random.shuffle(data)
    # separating the now reshuffled images from their classification.
    x, y = data[:, :-1], data[:, -1]
    return x, y


if __name__ == "__main__":
    main()
