import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms


def get_cifar_loaders(batch_size, num_workers):
    """Get the `DataLoader`s for this experiment.

    Args:
        batch_size: Batch size for each `DataLoader`.
        num_workers: Number of worker threads for each `DataLoader`.

    Returns:
        train_loader, test_loader, classes: Data loaders and a tuple of valid classes.
    """
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Load CIFAR10 dataset
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=train_transform)
    train_loader = data.DataLoader(train_set, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=test_transform)
    test_loader = data.DataLoader(test_set, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, classes
