import abc

import torch
import torchvision.transforms as tvt
from pytorch_ood.dataset.img import LSUNResize, Textures, TinyImageNetResize
from pytorch_ood.utils import ToRGB, ToUnknown
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10


class Detector(abc.ABC):
    """From https://github.com/kkirchheim/pytorch-ood"""

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        Forwards to predict
        """
        return self.predict(*args, **kwargs)

    @abc.abstractmethod
    def fit(self, data_loader: DataLoader) -> "Detector":
        """
        Fit the detector to a dataset. Some methods require this.

        param data_loader: dataset to fit on. This is usually the training dataset.

        raise ModelNotSetException: if model was not set
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_features(self, x: torch.Tensor, y: torch.Tensor) -> "Detector":
        """
        Fit the detector directly on features. Some methods require this.

        param x: training features to use for fitting.
        param y: corresponding class labels.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates outlier scores. Inputs will be passed through the model.

        param x: batch of data
        :return: outlier scores for points

        raise RequiresFitException: if detector has to be fitted to some data
        raise ModelNotSetException: if model was not set
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates outlier scores based on features.

        param x: batch of data
        :return: outlier scores for points

        raise RequiresFitException: if detector has to be fitted to some data
        """
        raise NotImplementedError


import warnings
from typing import Callable, List, Optional, TypeVar

from pytorch_ood.utils import (TensorBuffer, contains_unknown,
                               extract_features, is_known, is_unknown)
from torch.autograd import Variable


class Mahalanobis(Detector):
    """
    Implements the Mahalanobis Method from the paper *A Simple Unified Framework for Detecting
    Out-of-Distribution Samples and Adversarial Attacks*.

    This method calculates a class center :math:`\\mu_y` for each class,
    and a shared covariance matrix :math:`\\Sigma` from the data.

    Also uses ODIN preprocessing.

    :see Implementation: `GitHub <https://github.com/pokaxpoka/deep_Mahalanobis_detector>`__
    :see Paper: `ArXiv <https://arxiv.org/abs/1807.03888>`__
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        eps: float = 0.002,
        norm_std: Optional[list] = None,
    ):
        """
        :param model: the Neural Network, should output features
        :param eps: magnitude for gradient based input preprocessing
        :param norm_std: Standard deviations for input normalization
        """
        super().__init__()
        self.model = model
        self.mu: Optional[torch.Tensor] = None
        self.cov: Optional[torch.Tensor] = None
        self.precision: Optional[torch.Tensor] = None
        self.eps: float = eps
        self.norm_std = norm_std

    def fit(self, data_loader: DataLoader, device: str = None) -> "Mahalanobis":
        """
        Fit parameters of the multi variate gaussian.

        :param data_loader: dataset to fit on.
        :param device: device to use
        :return:
        """
        if device is None:
            device = list(self.model.parameters())[0].device
            print(f"No device given. Will use '{device}'.")

        z, y = extract_features(data_loader, self.model, device)
        return self.fit_features(z, y, device)

    def fit_features(
        self, z: torch.Tensor, y: torch.Tensor, device: str = None
    ) -> "Mahalanobis":
        """
        Fit parameters of the multi variate gaussian.

        :param z: features
        :param y: class labels
        :param device: device to use
        :return:
        """

        if device is None:
            device = z.device
            print(f"No device given. Will use '{device}'.")

        z, y = z.to(device), y.to(device)

        print("Calculating mahalanobis parameters.")
        classes = y.unique()

        # we assume here that all class 0 >= labels <= classes.max() exist
        assert len(classes) == classes.max().item() + 1
        assert not contains_unknown(classes)

        n_classes = len(classes)
        self.mu = torch.zeros(size=(n_classes, z.shape[-1])).to(device)
        self.cov = torch.zeros(size=(z.shape[-1], z.shape[-1])).to(device)

        for clazz in range(n_classes):
            idxs = y.eq(clazz)
            assert idxs.sum() != 0
            zs = z[idxs].to(device)
            self.mu[clazz] = zs.mean(dim=0)
            self.cov += (zs - self.mu[clazz]).T.mm(zs - self.mu[clazz])

        self.cov += torch.eye(self.cov.shape[0], device=self.cov.device) * 1e-6
        self.precision = torch.linalg.inv(self.cov)
        return self

    def predict_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates mahalanobis distance directly on features.
        ODIN preprocessing will not be applied.

        :param x: features, as given by the model.
        """
        features = x.view(x.size(0), x.size(1), -1)
        features = torch.mean(features, 2)
        noise_gaussian_scores = []

        for clazz in range(self.n_classes):
            centered_features = features.data - self.mu[clazz]
            term_gau = (
                -0.5
                * torch.mm(
                    torch.mm(centered_features, self.precision), centered_features.t()
                ).diag()
            )
            noise_gaussian_scores.append(term_gau.view(-1, 1))

        noise_gaussian_score = torch.cat(noise_gaussian_scores, 1)
        noise_gaussian_score = torch.max(noise_gaussian_score, dim=1).values

        return -noise_gaussian_score

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor
        """
        if self.eps > 0:
            x = self._odin_preprocess(x, x.device)

        features = self.model(x)
        return self.predict_features(features)

    def _odin_preprocess(self, x: torch.Tensor, dev: str):
        """
        NOTE: the original implementation uses mean over feature maps. here, we just flatten
        """
        # does not work in inference mode, this sometimes collides with pytorch-lightning
        if torch.is_inference_mode_enabled():
            warnings.warn(
                "ODIN not compatible with inference mode. Will be deactivated."
            )

        with torch.inference_mode(False):
            if torch.is_inference(x):
                x = x.clone()

            with torch.enable_grad():
                x = Variable(x, requires_grad=True)
                features = self.model(x)
                features = features.view(features.shape[0], -1)  # flatten
                score = None

                for clazz in range(self.n_classes):
                    centered_features = features.data - self.mu[clazz]
                    term_gau = (
                        -0.5
                        * torch.mm(
                            torch.mm(centered_features, self.precision),
                            centered_features.t(),
                        ).diag()
                    )

                    if clazz == 0:
                        score = term_gau.view(-1, 1)
                    else:
                        score = torch.cat((score, term_gau.view(-1, 1)), dim=1)

                # calculate gradient of inputs with respect to score of predicted class,
                # according to mahalanobis distance
                sample_pred = score.max(dim=1).indices
                batch_sample_mean = self.mu.index_select(0, sample_pred)
                centered_features = features - Variable(batch_sample_mean)
                pure_gau = (
                    -0.5
                    * torch.mm(
                        torch.mm(centered_features, Variable(self.precision)),
                        centered_features.t(),
                    ).diag()
                )
                loss = torch.mean(-pure_gau)
                loss.backward()

                gradient = torch.sign(x.grad.data)

        if self.norm_std:
            for i, std in enumerate(self.norm_std):
                gradient.index_copy_(
                    1,
                    torch.LongTensor([i]).to(dev),
                    gradient.index_select(1, torch.LongTensor([i]).to(dev)) / std,
                )
        perturbed_x = x.data - self.eps * gradient

        return perturbed_x

    @property
    def n_classes(self):
        return self.mu.shape[0]


def get_ood_datasets():
    # Setup preprocessing
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    trans = tvt.Compose(
        [
            tvt.Resize(size=(32, 32)),
            ToRGB(),
            tvt.ToTensor(),
            tvt.Normalize(std=std, mean=mean),
        ]
    )

    # Setup datasets
    dataset_in_test = CIFAR10(root="data", train=False, transform=trans, download=True)

    # create all OOD datasets
    ood_datasets = [Textures, TinyImageNetResize, LSUNResize]
    datasets = {}
    for ood_dataset in ood_datasets:
        dataset_out_test = ood_dataset(
            root="data", transform=trans, target_transform=ToUnknown(), download=True
        )
        test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=256)
        datasets[ood_dataset.__name__] = test_loader

    # %%
