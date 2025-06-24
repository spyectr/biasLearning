from torchvision import datasets, transforms
from emnist import extract_training_samples, extract_test_samples
# from afromnist import EthiopicMNIST, VaiMNIST, OsmanyaMNIST
# from kannada_mnist import KannadaMNIST
from torchvision.datasets import KMNIST, FashionMNIST, QMNIST, SVHN
from sklearn.datasets import fetch_openml

transform = transforms.Compose([transforms.Resize((10,10)), transforms.ToTensor()])

# TorchVision sets
_ = datasets.MNIST    ('.', train=True,  download=True, transform=transform)
_ = datasets.MNIST    ('.', train=False, download=True, transform=transform)
_ = FashionMNIST      ('.', train=True,  download=True, transform=transform)
_ = FashionMNIST      ('.', train=False, download=True, transform=transform)
_ = KMNIST            ('.', train=True,  download=True, transform=transform)
_ = KMNIST            ('.', train=False, download=True, transform=transform)
_ = QMNIST            ('.', train=True,  download=True, transform=transform)
_ = QMNIST            ('.', train=False, download=True, transform=transform)
_ = SVHN              ('.', split='train', download=True, transform=transform)
_ = SVHN              ('.', split='test',  download=True, transform=transform)

# EMNIST via emnist package
_ = extract_training_samples('digits')
_ = extract_test_samples   ('digits')

# Afro-MNIST
# _ = EthiopicMNIST('.'); _ = VaiMNIST('.'); _ = OsmanyaMNIST('.')

# Kannada-MNIST
# _ = KannadaMNIST('.')

# USPS & Semeion via openml (will cache in ~/scikit_learn_data)
fetch_openml('usps',    version=1)
fetch_openml('semeion', version=1)