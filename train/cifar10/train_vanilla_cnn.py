import yaml
import torch
from tqdm import tqdm

from datasets.cifar10 import CroppedCIFAR10
from models.cnn import VanillaCNN
from models.model_utils import get_n_params
import torchvision.transforms as transforms
import torch.optim as optim

CFG_PATH = "./trn_cfg.yaml"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_datasets(dataset_cfg: dict):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = CroppedCIFAR10(
        root=dataset_cfg["dir"],
        train=True,
        download=True,
        transform=transform,
        crop_width=dataset_cfg['crop_width'],
        crop_height=dataset_cfg['crop_height']
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=dataset_cfg["batch_size"],
        shuffle=True,
        num_workers=dataset_cfg['num_workers']
    )
    testset = CroppedCIFAR10(
        root=dataset_cfg["dir"],
        train=False,
        download=True,
        transform=transform,
        crop_width=dataset_cfg['crop_width'],
        crop_height=dataset_cfg['crop_height']
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=dataset_cfg["test_batch_size"],
        shuffle=False,
        num_workers=dataset_cfg['num_workers']
    )

    return trainloader, testloader


def train(optim_cfg: dict, trainloader: torch.utils.data.DataLoader):

    model = VanillaCNN.build(DEVICE)
    print(f"Num params: {get_n_params(model)}")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=optim_cfg['lr'], momentum=0.9)


    for epoch in range(optim_cfg["num_epochs"]):  # loop over the dataset multiple times

        running_loss = 0.0
        pbar = tqdm(enumerate(trainloader, 0), total=len(trainloader))
        for i, data in pbar:
            # get the inputs; data is a list of [inputs, labels]
            crops, imgs, labels = data

            crops = crops.to(DEVICE)
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(crops)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pbar.set_description("Loss: {:.5f}".format(loss.item()))
            running_loss += loss.item()
        print("Loss after epoch {}: {:.4f}".format(epoch + 1, running_loss / len(trainloader)))

    print('Finished Training')
    return model

def eval(model: torch.nn.Module, testloader: torch.utils.data.DataLoader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in tqdm(testloader, total=len(testloader)):
            crops, imgs, labels = data
            crops = crops.to(DEVICE)
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            # calculate outputs by running images through the network
            outputs = model(crops)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc



if __name__ == "__main__":
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    trainloader, testloader = get_datasets(cfg['dataset'])

    model = train(cfg['optimizer'], trainloader)
    acc = eval(model, testloader)

    print("Accuracy: {:.4f}".format(acc))