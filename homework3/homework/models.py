from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """

        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        features = self.features(z)
        
        logits = self.classifier(features)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)

class DownBlock(nn.Module):
    """
    Downsampling block for the encoder path of the U-Net architecture
    """
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_batchnorm = use_batchnorm
        
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x)
        
        features = x
        
        x = self.pool(x)
        
        return x, features

class UpBlock(nn.Module):
    """
    Upsampling block for the decoder path of the U-Net architecture
    """
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.use_batchnorm = use_batchnorm
        
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, skip_features):
        x = self.up(x)
        
        diffY = skip_features.size()[2] - x.size()[2]
        diffX = skip_features.size()[3] - x.size()[3]
        
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([skip_features, x], dim=1)
        
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x)
        
        return x

class Detector(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 16,
        depth: int = 4,
        use_batchnorm: bool = True,
    ):
        """
        A single model that performs segmentation and depth regression
        
        Args:
            in_channels: int, number of input channels
            num_classes: int, number of segmentation classes
            base_channels: int, number of base channels (will be doubled with each level)
            depth: int, depth of the U-Net architecture
            use_batchnorm: bool, whether to use batch normalization
        """
        super().__init__()
        
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))
        
        self.num_classes = num_classes
        self.depth = depth
        
        self.down_blocks = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(depth):
            out_channels = base_channels * (2 ** i)
            self.down_blocks.append(DownBlock(current_channels, out_channels, use_batchnorm))
            current_channels = out_channels
        
        self.middle_conv1 = nn.Conv2d(current_channels, current_channels * 2, kernel_size=3, padding=1)
        self.middle_conv2 = nn.Conv2d(current_channels * 2, current_channels, kernel_size=3, padding=1)
        
        if use_batchnorm:
            self.middle_bn1 = nn.BatchNorm2d(current_channels * 2)
            self.middle_bn2 = nn.BatchNorm2d(current_channels)
        
        self.up_blocks = nn.ModuleList()
        
        for i in reversed(range(depth)):
            in_ch = current_channels
            out_ch = base_channels * (2 ** i)
            self.up_blocks.append(UpBlock(in_ch, out_ch, use_batchnorm))
            current_channels = out_ch
        
        self.final_seg = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        self.final_depth = nn.Conv2d(base_channels, 1, kernel_size=1)
        
        self.use_batchnorm = use_batchnorm

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.
        
        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]
        
        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        skip_features = []
        for down_block in self.down_blocks:
            z, features = down_block(z)
            skip_features.append(features)
        
        z = self.middle_conv1(z)
        if self.use_batchnorm:
            z = self.middle_bn1(z)
        z = F.relu(z)
        
        z = self.middle_conv2(z)
        if self.use_batchnorm:
            z = self.middle_bn2(z)
        z = F.relu(z)
        
        for up_block, skip in zip(self.up_blocks, reversed(skip_features)):
            z = up_block(z, skip)
        
        seg_logits = self.final_seg(z)
        depth = self.final_depth(z).squeeze(1)
        
        return seg_logits, depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).
        
        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]
        
        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)
        
        depth = torch.sigmoid(raw_depth)
        
        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
