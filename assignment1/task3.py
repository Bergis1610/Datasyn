import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    """ 1. find index of max element in out-vector
        2. compare that to the index of the 1 in the target vector
            if they are the same, +1, if not +0 
    """
    correct_guesses = 0.0
    out = model.forward(X)

    for x, y in zip(out, targets): 
        ix = np.argmax(x)
        iy = np.argmax(y)

        if ix == iy :
            correct_guesses += 1

    accuracy = correct_guesses/X.shape[0]
    
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """


        # TODO: Implement this function (task 3b)

        out = self.model.forward(X_batch)
        self.model.backward(X_batch, out, Y_batch)
        self.model.w -= self.model.grad * self.learning_rate
        loss = cross_entropy_loss(Y_batch, out)

        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.
    
    print("------------------- task 3b and 3c ------------------------- \n")
    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .6])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    print("--------------------------------------------------------- \n")

    print("-------------------- task 4b ---------------------------- \n")

    # Intialize model with lambda = 0
    model_0 = SoftmaxModel(l2_reg_lambda= 0.0)
    # Train model
    trainer = SoftmaxTrainer(
        model_0, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_0, val_history_0 = trainer.train(num_epochs)
    # Intialize model
    model_1 = SoftmaxModel(l2_reg_lambda= 1.0)
    # Train model
    trainer = SoftmaxTrainer(
        model_1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_1, val_history_1 = trainer.train(num_epochs)

    lam0_img0 = model_0.w[:,0][:-1].reshape(28, 28)
    lam0_img1 = model_0.w[:,1][:-1].reshape(28, 28)
    lam0_img2 = model_0.w[:,2][:-1].reshape(28, 28)
    lam0_img3 = model_0.w[:,3][:-1].reshape(28, 28)
    lam0_img4 = model_0.w[:,4][:-1].reshape(28, 28)
    lam0_img5 = model_0.w[:,5][:-1].reshape(28, 28)
    lam0_img6 = model_0.w[:,6][:-1].reshape(28, 28)
    lam0_img7 = model_0.w[:,7][:-1].reshape(28, 28)
    lam0_img8 = model_0.w[:,8][:-1].reshape(28, 28)
    lam0_img9 = model_0.w[:,9][:-1].reshape(28, 28)

    plt.imsave("lam0_0.png", lam0_img0, cmap="gray")
    plt.imsave("lam0_1.png", lam0_img1, cmap="gray")
    plt.imsave("lam0_2.png", lam0_img2, cmap="gray")
    plt.imsave("lam0_3.png", lam0_img3, cmap="gray")
    plt.imsave("lam0_4.png", lam0_img4, cmap="gray")
    plt.imsave("lam0_5.png", lam0_img5, cmap="gray")
    plt.imsave("lam0_6.png", lam0_img6, cmap="gray")
    plt.imsave("lam0_7.png", lam0_img7, cmap="gray")
    plt.imsave("lam0_8.png", lam0_img8, cmap="gray")
    plt.imsave("lam0_9.png", lam0_img9, cmap="gray")

    lam1_img0 = model_1.w[:,0][:-1].reshape(28, 28)
    lam1_img1 = model_1.w[:,1][:-1].reshape(28, 28)
    lam1_img2 = model_1.w[:,2][:-1].reshape(28, 28)
    lam1_img3 = model_1.w[:,3][:-1].reshape(28, 28)
    lam1_img4 = model_1.w[:,4][:-1].reshape(28, 28)
    lam1_img5 = model_1.w[:,5][:-1].reshape(28, 28)
    lam1_img6 = model_1.w[:,6][:-1].reshape(28, 28)
    lam1_img7 = model_1.w[:,7][:-1].reshape(28, 28)
    lam1_img8 = model_1.w[:,8][:-1].reshape(28, 28)
    lam1_img9 = model_1.w[:,9][:-1].reshape(28, 28)

    plt.imsave("lam1_0.png", lam1_img0, cmap="gray")
    plt.imsave("lam1_1.png", lam1_img1, cmap="gray")
    plt.imsave("lam1_2.png", lam1_img2, cmap="gray")
    plt.imsave("lam1_3.png", lam1_img3, cmap="gray")
    plt.imsave("lam1_4.png", lam1_img4, cmap="gray")
    plt.imsave("lam1_5.png", lam1_img5, cmap="gray")
    plt.imsave("lam1_6.png", lam1_img6, cmap="gray")
    plt.imsave("lam1_7.png", lam1_img7, cmap="gray")
    plt.imsave("lam1_8.png", lam1_img8, cmap="gray")
    plt.imsave("lam1_9.png", lam1_img9, cmap="gray")

    print("--------------------------------------------------------- \n")



    print("-------------------- task 4c ---------------------------- \n")
    l2_lambdas = [1, 0.1, 0.01, 0.001]

    model1 = SoftmaxModel(l2_reg_lambda = 1)

    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg1, val_history_reg1 = trainer.train(num_epochs)

    
    model2 = SoftmaxModel(l2_reg_lambda = 0.1)

    trainer = SoftmaxTrainer(
        model2, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg2, val_history_reg2 = trainer.train(num_epochs)

    
    model3 = SoftmaxModel(l2_reg_lambda = 0.01)

    trainer = SoftmaxTrainer(
        model3, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg3, val_history_reg3 = trainer.train(num_epochs)


    model4 = SoftmaxModel(l2_reg_lambda = 0.001)

    trainer = SoftmaxTrainer(
        model4, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg4, val_history_reg4 = trainer.train(num_epochs)

    utils.plot_loss(val_history_reg1["accuracy"], "Validation Accuracy lambda=1.0")
    utils.plot_loss(val_history_reg2["accuracy"], "Validation Accuracy lambda=0.1")
    utils.plot_loss(val_history_reg3["accuracy"], "Validation Accuracy lambda=0.01")
    utils.plot_loss(val_history_reg4["accuracy"], "Validation Accuracy lambda=0.001")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()


    


    print("--------------------------------------------------------- \n")




    print("-------------------- task 4d ---------------------------- \n")








    print("--------------------------------------------------------- \n")
    print("-------------------- task 4e ---------------------------- \n")

    norm1 = np.linalg.norm(model1.w)
    norm2 = np.linalg.norm(model2.w)
    norm3 = np.linalg.norm(model3.w)
    norm4 = np.linalg.norm(model4.w)

    plt.plot(l2_lambdas[0], norm1, marker = "o")
    plt.plot(l2_lambdas[1], norm2, marker = "o")
    plt.plot(l2_lambdas[2], norm3, marker = "o")
    plt.plot(l2_lambdas[3], norm4, marker = "o")
    plt.xlabel("lambda values")
    plt.ylabel("L2 norm")
    plt.legend()
    plt.savefig("task4d_l2_reg_norms.png")
    plt.show()






    print("--------------------------------------------------------- \n")


if __name__ == "__main__":
    main()
