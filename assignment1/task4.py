import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
from task3 import calculate_accuracy, SoftmaxTrainer 
np.random.seed(0)





def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda_model1 = 0.0
    l2_reg_lambda_model2 = 1.0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize models
    model1 = SoftmaxModel(l2_reg_lambda_model1)
    model2 = SoftmaxModel(l2_reg_lambda_model2)

    # Train model
    trainer1 = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    trainer2 = SoftmaxTrainer(
        model2, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_model1, val_history_model1 = trainer1.train(num_epochs)
    train_history_model2, val_history_model2 = trainer2.train(num_epochs)

    print("\n")
    print("---------------------------------------------------------------------------------------------------------")
    print("\n")

    print("Final Train Cross Entropy Loss for model 1:",
          cross_entropy_loss(Y_train, model1.forward(X_train)))
    print("Final Validation Cross Entropy Loss for model 1:",
          cross_entropy_loss(Y_val, model1.forward(X_val)))
    print("Final Train accuracy for model 1:", calculate_accuracy(X_train, Y_train, model1))
    print("Final Validation accuracy for model 1:", calculate_accuracy(X_val, Y_val, model1))

    print("\n")
    print("---------------------------------------------------------------------------------------------------------")
    print("\n")

    print("Final Train Cross Entropy Loss model 2:",
          cross_entropy_loss(Y_train, model2.forward(X_train)))
    print("Final Validation Cross Entropy Loss model 2:",
          cross_entropy_loss(Y_val, model2.forward(X_val)))
    print("Final Train accuracy model 2:", calculate_accuracy(X_train, Y_train, model2))
    print("Final Validation accuracy model 2:", calculate_accuracy(X_val, Y_val, model2))

    print("\n")
    print("---------------------------------------------------------------------------------------------------------")
    print("\n")

    plt.ylim([0.2, .8])
    utils.plot_loss(train_history_model1["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history_model1["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task4_softmax_train_loss_model1.png")
    plt.show()

    

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history_model1["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history_model1["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4_softmax_train_accuracy_model1.png")
    plt.show()

    print("\n")
    print("---------------------------------------------------------------------------------------------------------")
    print("\n")

    plt.ylim([0.2, .8])
    utils.plot_loss(train_history_model2["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history_model2["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task4_softmax_train_loss_model2.png")
    plt.show()

    
    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history_model2["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history_model2["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4_softmax_train_accuracy_model2.png")
    plt.show()

    print("\n")
    print("---------------------------------------------------------------------------------------------------------")
    print("\n")

    # Train a model with L2 regularization (task 4b)

    model1 = SoftmaxModel(l2_reg_lambda=1.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
    # You can finish the rest of task 4 below this point.

    # Plotting of softmax weights (Task 4b)
    plt.imsave("task4b_softmax_weight.png", model1.w, cmap="gray")




    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [1, .1, .01, .001]
    plt.savefig("task4c_l2_reg_accuracy.png")

    # Task 4d - Plotting of the l2 norm for each weight

    plt.savefig("task4d_l2_reg_norms.png")


if __name__ == "__main__":
    main()


