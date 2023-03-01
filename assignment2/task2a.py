import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] in the range (-1, 1)
    """

    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    new_X = np.zeros((X.shape[0], X.shape[1]+1))
    training_data, _, _, _ = utils.load_full_mnist()
    mean = np.mean(training_data)
    std = np.std(training_data)
    # Printing
    #print("Mean: " + str(mean))
    #print("Standard deviation: " + str(std))

    for i, batch in enumerate(X):
        for j, pix in enumerate(batch):
            new_X[i, j] = (pix-mean)/std

        new_X[i, -1] = 1

    return new_X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """

    summ = - np.sum(targets*np.log(outputs), axis=1)

    mean = np.mean(summ)

    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    return mean


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3b hyperparameter
                 use_improved_weight_init: bool,  # Task 3a hyperparameter
                 use_relu: bool  # Task 4 hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_relu = use_relu
        self.use_improved_weight_init = use_improved_weight_init
        self.hidden_layer_output = []

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)

            if use_improved_weight_init:
                w = np.random.normal(0, 1/np.sqrt(w_shape[0]), w_shape)
            else:
                w = np.random.uniform(-1, 1, w_shape)

            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

        self.zs = []
        self.activations = []

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def improved_sigmoid(self, z):
        return 1.7159 * np.tanh(2/3 * z)

    def improved_sigmoid_prime(self, z):
        return 1.14393 * 1/((np.cosh(2*z/3))**2)

    def softMax(self, z):
        return np.exp(z)/(np.sum(np.exp(z), axis=1)[:, None])

    def activation(self, layerIndex, z):
        if (layerIndex != len(self.ws)-1):

            return self.sigmoid(z)

        else:

            return self.softMax(z)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...

        self.hidden_layer_output = []
        self.zs = []
        self.number_of_layers = len(self.ws)

        logit = X
        self.hidden_layer_output.append(X)

        for i in range(self.number_of_layers-1):
            z = logit@self.ws[i]

            if self.use_improved_sigmoid:
                logit = self.improved_sigmoid(z)
            else:
                logit = self.sigmoid(z)

            self.zs.append(z)
            self.hidden_layer_output.append(logit)

        z = logit@self.ws[-1]

        return self.softMax(z)

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        self.grads = []

        del_k = -(targets - outputs)
        self.grads = []

        self.grads.append((self.hidden_layer_output[-1].T@del_k) / X.shape[0])
        del_i = del_k
        for i in range(1, self.number_of_layers):

            if (self.use_improved_sigmoid):
                del_i = (del_i @ self.ws[-i].T) * \
                    self.improved_sigmoid_prime(self.zs[-i])
            else:
                del_i = (del_i @ self.ws[-i].T) * \
                    self.sigmoid_prime(self.zs[-i])

            self.grads.append(
                (self.hidden_layer_output[-i-1].T @ del_i)/X.shape[0])

        self.grads.reverse()

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """

    new_Y = []
    for y in Y:
        vector = [0 for i in range(num_classes)]

        vector[y[0]] = 1
        new_Y.append(vector)

    return np.array(new_Y)


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**1,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = True  # False
    use_improved_weight_init = True  # False
    use_relu = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    main()
