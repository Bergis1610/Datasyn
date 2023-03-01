import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    #Standard model
    num_epochs = 50
    learning_rate = .01
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True  # False
    use_improved_weight_init = True  # False
    use_momentum = True  # False
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Task 4f)
    use_improved_sigmoid = False
    use_relu = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model_10 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_10 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_10, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_10, val_history_10 = trainer_10.train(num_epochs)

    #Task 4e)
    """ num_epochs = 50
    learning_rate = .01
    batch_size = 32
    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True  # False
    use_improved_weight_init = True  # False
    use_momentum = True  # False
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model_10 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_10 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_10, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_10, val_history_10 = trainer_10.train(num_epochs) """

    # Task 4d)
    """ num_epochs = 50
    learning_rate = .01
    batch_size = 32
    neurons_per_layer = [60, 60, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True  # False
    use_improved_weight_init = True  # False
    use_momentum = True  # False
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs) """

    # Task 4a) and 4b)
    """ num_epochs = 50
    learning_rate = .01
    batch_size = 32
    neurons_per_layer = [128, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True  # False
    use_improved_weight_init = True  # False
    use_momentum = True  # False
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model_32 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_32 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_32, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_32, val_history_32 = trainer_32.train(num_epochs) """

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    # Task 3)
    """ use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False
    use_relu = False
    shuffle_data = False

    # Train a new model with new parameters
    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_no_shuffle, val_history_no_shuffle = trainer_shuffle.train(
        num_epochs) """

    """ use_improved_weight_init = True
    use_improved_sigmoid = False
    use_momentum = False
    use_relu = False
    shuffle_data = False

    # Train a new model with new parameters
    model_weight = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_weight = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_weight, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_weight, val_history_weight = trainer_weight.train(
        num_epochs) """

    """ use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = False
    use_relu = False
    shuffle_data = False

    # Train a new model with new parameters
    model_sigmoid = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_sigmoid = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_sigmoid, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_sigmoid, val_history_sigmoid = trainer_sigmoid.train(
        num_epochs) """

    """
    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = True
    use_relu = False
    shuffle_data = False

    # Train a new model with new parameters
    model_momentum = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_momentum = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_momentum, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_momentum, val_history_momentum = trainer_momentum.train(
        num_epochs) """

    plt.subplot(1, 2, 1)
    # Task 4f)
    utils.plot_loss(train_history["loss"],
                    "Training loss sigmoid", npoints_to_average=10)
    utils.plot_loss(train_history_10["loss"],
                    "Training loss relu", npoints_to_average=10)

    # Task 4e)
    #utils.plot_loss(train_history["loss"],"Training loss standard", npoints_to_average=10)
    #utils.plot_loss(train_history_10["loss"], "Training loss 10 layers", npoints_to_average=10)

    # Task 4d)
    #utils.plot_loss(train_history["loss"],"Training loss", npoints_to_average=10)
    #utils.plot_loss(val_history["loss"],"Validation loss")
    # Task 4a) and 4b)
    #utils.plot_loss(train_history_32["loss"],"Task 3 Model 128", npoints_to_average=10)

    # Task 3)
    #utils.plot_loss(train_history_weight["loss"], "Task 2 Model - with weights", npoints_to_average=10)
    #utils.plot_loss(train_history_sigmoid["loss"], "Task 2 Model - with sigmoid", npoints_to_average=10)
    #utils.plot_loss(train_history_momentum["loss"], "Task 2 Model - with momentum", npoints_to_average=10)
    plt.ylabel("Loss")
    plt.xlabel("Number of training steps")
    #utils.plot_loss(train_history_no_shuffle["loss"], "Task 2 Model - No dataset shuffling", npoints_to_average=10)
    plt.ylim([0, .4])

    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .99])

    # Task 4f)
    utils.plot_loss(val_history["accuracy"], "Validation accuracy sigmoid")
    utils.plot_loss(val_history_10["accuracy"], "Validation accuracy relu")

    # Task 4e)
    #utils.plot_loss(val_history["accuracy"], "Validation accuracy standard")
    #utils.plot_loss(val_history_10["accuracy"],"Validation accuracy 10 layers")

    # Task 4d)
    #utils.plot_loss(train_history["accuracy"], "Training accuracy")
    #utils.plot_loss(val_history["accuracy"], "Validation accuracy")

    # Task 4a) and 4b)
    #utils.plot_loss(val_history_32["accuracy"], "Task 3 Model 128")

    # Task 3)
    #utils.plot_loss(val_history_weight["accuracy"], "Task 2 Model - with weight")
    #utils.plot_loss(val_history_sigmoid["accuracy"], "Task 2 Model - with sigmoid")
    #utils.plot_loss(val_history_momentum["accuracy"], "Task 2 Model - with momentum")

    #utils.plot_loss(val_history_no_shuffle["accuracy"], "Task 2 Model - No Dataset Shuffling")

    plt.ylabel("Accuracy")
    plt.xlabel("Number of training steps")
    plt.legend()
    plt.savefig("task3abc.png")
    plt.show()


if __name__ == "__main__":
    main()
