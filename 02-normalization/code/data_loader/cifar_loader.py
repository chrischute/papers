import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms


class CIFARLoader(data.DataLoader):
    """DataLoader for CIFAR-10."""
    def __init__(self, phase, batch_size, num_workers):
        self.phase = phase
        transforms_list = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)] if phase == 'train' else []
        transforms_list += [transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=transforms.Compose(transforms_list))
        shuffle = (phase == 'train')
        super(CIFARLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
