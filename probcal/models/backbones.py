import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torch import nn
from torchvision.models.mobilenet import mobilenet_v3_large
from torchvision.models.mobilenet import MobileNet_V3_Large_Weights
from transformers import BatchEncoding
from transformers import DistilBertModel
from transformers import ViTModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling


class Backbone(nn.Module):
    """Base class to ensure a common interface for all backbones.

    Attributes:
        output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
    """

    def __init__(self, output_dim: int = 64):
        super(Backbone, self).__init__()
        self.output_dim = output_dim


class MLP(Backbone):
    """An MLP feature extractor for (N, d) input data.

    Attributes:
        layers (nn.Sequential): The layers of this MLP.
    """

    def __init__(self, input_dim: int = 1, output_dim: int = 64):
        """Instantiate an MLP backbone.

        Args:
            input_dim (int, optional): Dimension of input feature vectors. Defaults to 1.
            output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
        """
        self.input_dim = input_dim
        super(MLP, self).__init__(output_dim=output_dim)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LargerMLP(Backbone):
    """A larger MLP feature extractor for (N, d) input data.

    Attributes:
        layers (nn.Sequential): The layers of this MLP.
    """

    def __init__(self, input_dim: int = 1, output_dim: int = 64):
        """Instantiate an MLP backbone.

        Args:
            input_dim (int, optional): Dimension of input feature vectors. Defaults to 1.
            output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
        """
        self.input_dim = input_dim
        super(LargerMLP, self).__init__(output_dim=output_dim)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ViT(Backbone):
    """A ViT feature extractor for images.

    Attributes:
        output_dim (int): Dimension of output feature vectors.
    """

    def __init__(self, output_dim: int = 64):
        """Initialize a ViT image feature extractor.

        Args:
            output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
        """
        super(ViT, self).__init__(output_dim=output_dim)
        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.projection_1 = nn.Linear(768, 384)
        self.projection_2 = nn.Linear(384, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs: BaseModelOutputWithPooling = self.backbone(pixel_values=x)
        h = outputs.pooler_output
        h = self.relu(self.projection_1(h))
        h = self.relu(self.projection_2(h))
        return h


class MobileNetV3(Backbone):
    """A MobileNetV3 feature extractor for 3x224x224 image tensors.

    Attributes:
        output_dim (int): Dimension of output feature vectors.
    """

    def __init__(self, output_dim: int = 64):
        """Initialize a MobileNetV3 feature extractor.

        Args:
            output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
        """
        super(MobileNetV3, self).__init__(output_dim=output_dim)

        self.backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).features
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten(start_dim=2)
        self.conv1d = nn.Conv1d(in_channels=960, out_channels=self.output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.flatten(self.avg_pool(self.backbone(x)))
        h = self.conv1d(h).squeeze(-1)
        return h


class DistilBert(Backbone):
    """A DistilBert feature extractor for text sequences."""

    def __init__(self, output_dim: int = 64, freeze_backbone: bool = False):
        """Initialize a DistilBert text feature extractor.

        Args:
            output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
            freeze_backbone (bool, optional): Whether to freeze the DistilBert backbone. Defaults to False.
        """
        super(DistilBert, self).__init__(output_dim=output_dim)
        self.backbone: DistilBertModel = DistilBertModel.from_pretrained("distilbert-base-cased")
        self.projection_1 = nn.Linear(768, 384)
        self.projection_2 = nn.Linear(384, self.output_dim)
        self.relu = nn.ReLU()
        if freeze_backbone:
            self._freeze_backbone(method="last_block")

    def forward(self, x: BatchEncoding) -> torch.Tensor:
        outputs: BaseModelOutput = self.backbone(**x)
        h = outputs.last_hidden_state[:, 0]
        h = self.relu(self.projection_1(h))
        h = self.relu(self.projection_2(h))
        return h

    def _freeze_backbone(self, method: str = "all") -> None:
        """Freeze the backbone during training based on the specified method.

        Args:
            method (str, optional): Method to selectively unfreeze parts of the backbone.
                Options are:
                        - 'all': Freeze all layers.
                        - 'last_block': Freeze all layers except the last transformer block.
                        - 'last_block_linear':  Freeze all layers except linear layers and layer norm in the last transformer block.

        Raises:
            ValueError: If an invalid method is provided.
        """
        # freeze all BERT layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        if method == "all":
            return

        elif method == "last_block":
            # manually unfreeze all layers in last transformer block
            for param in self.backbone.transformer.layer[-1].parameters():
                param.requires_grad = True
        elif method == "last_block_linear":
            # manually unfreeze only linear layers (and layer norm) in final transformer block
            for param in self.backbone.transformer.layer[-1].ffn.parameters():
                param.requires_grad = True
            self.backbone.transformer.layer[-1].output_layer_norm.weight.requires_grad = True
        else:
            ValueError(f"Invalid method: {method}")


class MNISTCNN(Backbone):
    """A CNN feature extractor for MNIST images (repeated along the channel dim to be 3x28x28)."""

    def __init__(self, output_dim: int = 64):
        """Initialize an MNISTCNN.

        Args:
            output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
        """
        super(MNISTCNN, self).__init__(output_dim=output_dim)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(5 * 5 * 64, output_dim * 2)
        self.fc2 = nn.Linear(output_dim * 2, output_dim)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.max_pool(self.relu(self.conv1(x))))
        x = self.bn2(self.max_pool(self.relu(self.conv2(x))))
        x = self.dropout(self.flat(x))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return x
