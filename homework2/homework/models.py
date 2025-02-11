"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return F.cross_entropy(logits, target)
        raise NotImplementedError("ClassificationLoss.forward() is not implemented")


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        input_features = 3 * h * w
        self.linear = nn.Linear(input_features, num_classes)

        # raise NotImplementedError("LinearClassifier.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """

        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        logits = self.linear(x_flat)
        return logits

        
        # raise NotImplementedError("LinearClassifier.forward() is not implemented")


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        input_features = 3 * h * w
        hidden_features = 128

        # I used AI for this layers part to help me understand
        self.layers = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, num_classes)
        )

        # raise NotImplementedError("MLPClassifier.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        return self.layers(x_flat)
        # raise NotImplementedError("MLPClassifier.forward() is not implemented")


class MLPClassifierDeep(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        num_layers: int = 4
    ):
        """
        An MLP with multiple hidden layers

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()
        input_features = 3 * h * w
        hidden_dim = 128
        
        # I got stuck here so I used AI to help me understand what to do here.
        layers = []
        layers.append(nn.Linear(input_features, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.layers = nn.Sequential(*layers)

        # raise NotImplementedError("MLPClassifierDeep.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        return self.layers(x_flat)
        # raise NotImplementedError("MLPClassifierDeep.forward() is not implemented")


# I used AI for this function
class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        num_layers: int = 4,
    ):
        """
        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()
        input_features = 3 * h * w
        hidden_dim = 128
        
        self.input_layer = nn.Linear(input_features, hidden_dim)
        
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers - 1)
        ])
        
        self.relu = nn.ReLU()
        
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        # raise NotImplementedError("MLPClassifierDeepResidual.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.relu(self.input_layer(x))
        
        for layer in self.hidden_layers:
            identity = x
            x = layer(x)
            x = x + identity  # Residual connection
            x = self.relu(x)
            
        x = self.output_layer(x)
        
        return x
        # raise NotImplementedError("MLPClassifierDeepResidual.forward() is not implemented")


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
