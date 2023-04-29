from flytekit import Resources, task

# import all dependencies
import torch
import numpy as np
from torchvision import transforms as T, datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import timm
from dataclasses_json import dataclass_json
from dataclasses import dataclass
import pathlib
import joblib


@dataclass_json
@dataclass
class Hyperparameters(object):
    """
    Class for the hyperparameters
    ...
    Attributes
    -----------
    epochs : int
        Number of Epochs
    lr : float = 0.0001
        Learning Rate
    batch_size : int
        Size of the batch
    model_name : str
        Name of the model to be imported from timm
        See more: https://github.com/huggingface/pytorch-image-models
    img_size : int
        Dimension of the input image
    train_path : str
        Filepath for training data
    """

    epochs: int = 20
    lr: float = 0.001
    batch_size: int = 16
    model_name: str = "tf_efficientnet_b4_ns"
    img_size: int = 224
    train_path: str = "data/train"


# instantiating Hyperparameter class
hp = Hyperparameters()

# declaring device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# getting model the accuracy
def accuracy(y_pred, y_true):
    """
    This function calculates the accuracy
    """
    y_pred = F.softmax(y_pred, dim=1)
    top_p, top_class = y_pred.topk(1, dim=1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))


# preparing dataset and loading data
def get_data(img_size, train_path, batch_size):
    """
    This function creates the train dataset using torchvision.datasets.ImageFolder
    and loads the data using DataLoader
    """
    # transorming image data
    train_transform = T.Compose(
        [
            T.Resize(size=(img_size, img_size)),
            T.RandomRotation(
                degrees=(-20, +20)
            ),
            T.ToTensor(), 
            T.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    # creating dataset
    
    trainset = datasets.ImageFolder((pathlib.Path(__file__).parent / train_path).resolve(), transform=train_transform)
    # loading data
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainloader


# creating model
def create_model(model_name):
    """
    This function imports the pre-trained image model from timm
    EfficientNet has 1000 output classes. This function also modifies
    the classifier of the model to have 2 output classes.
    """
    # importing model using timm
    model = timm.create_model(model_name, pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # updating model classifier to have 2 classes
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1792, out_features=625),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=2),
    )
    # moving model to device
    model.to(device)
    return model


# creating class for training model
class PneumoniaTrainer(nn.Module):
    def __init__(self, criterion=None, optimizer=None, schedular=None):
        self.criterion = criterion
        self.optimizer = optimizer
        self.schedular = schedular

    def train_batch_loop(self, model, trainloader):
        train_loss = 0.0
        train_acc = 0.0

        for idx, (images, labels) in enumerate(trainloader):
            # move the data to CPU
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy(logits, labels)

        return train_loss / len(trainloader), train_acc / len(trainloader)

    def fit(self, model, trainloader, epochs):
        for i in range(epochs):
            model.train()
            avg_train_loss, avg_train_acc = self.train_batch_loop(model, trainloader)
        return model


# fitting model
def fit_model(model, lr, trainloader, epochs):
    trainer = PneumoniaTrainer(criterion, optimizer)
    return trainer.fit(model, trainloader, epochs=epochs)


@task(requests=Resources(cpu="2",mem="4Gi",storage="0Gi",ephemeral_storage="0Gi"),limits=Resources(cpu="2",mem="4Gi",storage="0Gi",ephemeral_storage="0Gi"),retries=3)
def run_workflow(hp: Hyperparameters) -> PneumoniaTrainer:
    model = create_model(hp.model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainloader = get_data(
        img_size=hp.img_size, train_path=hp.train_path, batch_size=hp.batch_size
    )
    return fit_model(model, hp.lr, trainloader, hp.epochs)


if __name__ == "__main__":
    run_workflow(hp=hp)