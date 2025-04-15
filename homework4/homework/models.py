from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dim: int = 128,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        # Input features: left and right track points (each with x,y coordinates)
        input_dim = n_track * 2 * 2
        output_dim = n_waypoints * 2  # Output: waypoints with x,y coordinates
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.size(0)
        
        # Flatten the track inputs
        left_flat = track_left.reshape(batch_size, -1)
        right_flat = track_right.reshape(batch_size, -1)
        
        # Concatenate left and right track points
        track_features = torch.cat([left_flat, right_flat], dim=1)
        
        # Pass through MLP
        output = self.mlp(track_features)
        
        # Reshape to (batch_size, n_waypoints, 2)
        waypoints = output.reshape(batch_size, self.n_waypoints, 2)
        
        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model
        
        # Embedding for waypoint queries
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        
        # Projection layers for track points
        self.track_encoder_left = nn.Linear(2, d_model)
        self.track_encoder_right = nn.Linear(2, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection to get waypoints
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.size(0)
        
        # Get query embeddings for waypoints
        # Shape: (batch_size, n_waypoints, d_model)
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Encode track points
        left_features = self.track_encoder_left(track_left)  # (b, n_track, d_model)
        right_features = self.track_encoder_right(track_right)  # (b, n_track, d_model)
        
        # Concatenate left and right track features
        track_features = torch.cat([left_features, right_features], dim=1)  # (b, 2*n_track, d_model)
        
        # Apply transformer decoder
        # Shape: (batch_size, n_waypoints, d_model)
        decoder_output = self.transformer_decoder(
            tgt=queries,
            memory=track_features
        )
        
        # Project to get final waypoints
        # Shape: (batch_size, n_waypoints, 2)
        waypoints = self.output_proj(decoder_output)
        
        return waypoints

class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        
        # CNN backbone (similar to ResNet-like architecture)
        self.backbone = nn.Sequential(
            # Initial conv layer
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Residual-like blocks
            self._make_res_block(32, 64, stride=2),
            self._make_res_block(64, 128, stride=2),
            self._make_res_block(128, 256, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fully connected layers for waypoint prediction
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_waypoints * 2)
        )

    def _make_res_block(self, in_channels, out_channels, stride=1):
        layers = []
        
        # Conv block with downsampling
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Second conv block
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        batch_size = image.size(0)
        
        # Normalize input
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        # Extract features using CNN backbone
        features = self.backbone(x)
        
        # Predict waypoints
        waypoints_flat = self.fc(features)
        
        # Reshape to (batch_size, n_waypoints, 2)
        waypoints = waypoints_flat.view(batch_size, self.n_waypoints, 2)
        
        return waypoints

MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
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
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
