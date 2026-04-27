"""
Image encoders for CLIP model.
Supports ResNet50 and Vision Transformer (ViT) architectures.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Encoder(nn.Module):
    """
    ResNet50-based image encoder.
    Paper: "Deep Residual Learning for Image Recognition" (He et al., 2015)
    https://arxiv.org/abs/1512.03385
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # TODO: Get the feature dimension
        self.feature_dim = resnet.fc.in_features

        # TODO: Remove classification head
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images into embeddings.
        Args:
            images: Batch of images [batch_size, 3, H, W]
        Returns:
            features: Image features [batch_size, 2048]
        """
        # TODO: Extract features
        # Shape: [batch_size, 2048, 1, 1]
        features = self.resnet(images)

        # TODO: Reshape to [batch_size, 2048]
        features = features.flatten(start_dim=1)

        return features


class ViTEncoder(nn.Module):
    """
    Vision Transformer (ViT) based image encoder.
    Paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovskiy et al., 2020)
    https://arxiv.org/abs/2010.11929
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # Load pretrained Vision Transformer ViT-B/16
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        vit = models.vit_b_16(weights=weights)

        # TODO: Get the feature dimension
        self.feature_dim = vit.heads.head.in_features

        # TODO: Replace classification head with identity
        vit.heads.head = nn.Identity()

        self.vit = vit

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images into embeddings.
        Args:
            images: Batch of images [batch_size, 3, 224, 224]
        Returns:
            features: Image features [batch_size, 768]
        """
        # TODO: Extract features
        # We use the [CLS] token representation as the image feature vector
        # [CLS] token is the first token in the sequence
        # It always interacts with all other tokens, so it captures global image information
        # Shape: [batch_size, 768]
        features = self.vit(images)

        return features


class ImageEncoder(nn.Module):
    """
    CLIP Image Encoder that supports both ResNet50 and ViT backbones.
    Extracts features from images and projects them to the embedding space.
    """

    def __init__(
        self,
        encoder_type="vit",
        embed_dim=512,
        pretrained=True,
        trainable_image_blocks=0,
    ):
        super().__init__()
        if encoder_type == "resnet":
            self.encoder = ResNet50Encoder(pretrained)
        elif encoder_type == "vit":
            self.encoder = ViTEncoder(pretrained)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Freeze pre-trained encoder parameters by default.
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze only the top image blocks for lightweight adaptation.
        if trainable_image_blocks > 0:
            if encoder_type == "resnet":
                # ResNet structure in self.resnet: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4]
                block_indices = [7, 6, 5, 4]
                for block_idx in block_indices[:trainable_image_blocks]:
                    for param in self.encoder.resnet[block_idx].parameters():
                        param.requires_grad = True
            elif encoder_type == "vit":
                layers = self.encoder.vit.encoder.layers
                for layer in layers[-trainable_image_blocks:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        # TODO: Create a sequential network that maps encoder features to embed_dim
        # Architecture: Linear -> ReLU -> Linear
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, self.encoder.feature_dim),
            nn.ReLU(),
            nn.Linear(self.encoder.feature_dim, embed_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images and project to embedding space.
        Args:
            images: Batch of images [batch_size, 3, H, W]
        Returns:
            embeddings: Image embeddings [batch_size, embed_dim]
        """
        # TODO: Extract features
        features = self.encoder(images)
        # TODO: Project features to embedding space
        embeddings = self.projection(features)
        return embeddings
