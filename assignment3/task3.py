import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer
from trainer import compute_loss_and_accuracy


class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
           
        self.num_classes = num_classes
        # Define the convolutional layers

       
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d([2, 2], stride=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d([2, 2], stride=2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5, 
                stride=1,
                padding=2,
            ),
            
            nn.ReLU(),
            nn.MaxPool2d([2, 2], stride=2),
    
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 128*16*16
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)

        batch_size = x.shape[0]
        features = self.feature_extractor(x)        
        linear_features = features.view(batch_size, 2048) ## [batch_size, number of out]
        out = self.classifier(linear_features)
        


        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, "task3 added batch normalization horizontal and vertical flip with prop 0.5")

    train_loss, train_acc = compute_loss_and_accuracy(trainer.dataloader_train, model, torch.nn.CrossEntropyLoss())
    print("Average training loss: ", train_loss.item() ," Average training accuracy: ", train_acc.item())

    train_loss, train_acc = compute_loss_and_accuracy(trainer.dataloader_val, model, torch.nn.CrossEntropyLoss())
    print("Average validation loss: ", train_loss.item() ," Average validation accuracy: ", train_acc.item())

    train_loss, train_acc = compute_loss_and_accuracy(trainer.dataloader_test, model, torch.nn.CrossEntropyLoss())
    print("Average testing loss: ", train_loss.item() ," Average testing accuracy: ", train_acc.item())


if __name__ == "__main__":
    main()
