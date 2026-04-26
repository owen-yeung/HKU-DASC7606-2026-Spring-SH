"""
Main CLIP model combining image and text encoders.
Implements contrastive learning for joint vision-language understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.image_encoder import ImageEncoder
from model.text_encoder import TextEncoder, TextTokenizer


class CLIP(nn.Module):
    """
    CLIP model for contrastive vision-language learning.
    Learns to align image and text representations in a joint embedding space.
    """

    def __init__(
        self,
        encoder_type="vit",
        embed_dim=512,
        temperature=0.07,
        pretrained=True,
    ):
        super().__init__()
        # TODO: Image encoder (ResNet50 or ViT)
        self.image_encoder = ImageEncoder(
            encoder_type=encoder_type,
            embed_dim=embed_dim,
            pretrained=pretrained,
        )

        # TODO: Text tokenizer and encoder (RoBERTa-based)
        self.text_tokenizer = TextTokenizer()
        self.text_encoder = TextEncoder(embed_dim=embed_dim, pretrained=pretrained)

        # Temperature parameter for scaling logits in contrastive loss
        # Learnable log-temperature
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())

    @property
    def temperature(self):
        # Return the actual temperature by exponentiating the log-temperature
        return self.log_temperature.exp()

    def compute_similarity(self, images, texts):
        """
        Compute cosine similarity between images and texts.

        Args:
            images: Batch of images [batch_size, 3, 224, 224]
            texts: List of text prompts [batch_size]
        Returns:
            logits: Similarity scores [batch_size, batch_size]
        """
        # TODO: Encode images
        # Shape: [batch_size, embed_dim]
        image_embeddings = self.image_encoder(images)

        # TODO: Tokenize text then encode tokens
        # Shape: [batch_size, embed_dim]
        input_ids, attention_mask = self.text_tokenizer(texts, device=images.device)
        text_embeddings = self.text_encoder(input_ids, attention_mask)

        # TODO: L2 normalize both embeddings, so that their dot product equals cosine similarity
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # TODO: Compute NxN similarity matrix, then scale by temperature
        # Shape: [batch_size, batch_size]
        logits = image_embeddings @ text_embeddings.t()
        logits = logits / self.temperature

        return logits

    def forward(self, images, texts, labels):
        """
        Compute symmetric contrastive loss.
        See: https://lilianweng.github.io/posts/2021-05-31-contrastive/#clip

        This loss encourages:
        - High similarity between matching image-text pairs
        - Low similarity between non-matching pairs

        Args:
            images: Batch of images [batch_size, 3, 224, 224]
            texts: List of text prompts [batch_size]
            labels: Diagonal labels indicating correct matches [batch_size]

        Returns:
            dict with keys:
            - "logits": Similarity scores [batch_size, batch_size]
            - "loss": Scalar contrastive loss
        """
        # TODO: Compute the NxN similarity logits
        # Shape: [batch_size, batch_size]
        logits = self.compute_similarity(images, texts)

        # TODO: Compute image-to-text cross entropy loss
        loss_i2t = F.cross_entropy(logits, labels)

        # TODO: Compute text-to-image cross entropy loss
        loss_t2i = F.cross_entropy(logits.t(), labels)

        # TODO: Average both loss directions
        loss = 0.5 * (loss_i2t + loss_t2i)

        return {"logits": logits, "loss": loss}

    def predict(self, images, texts):
        """
        Predict class labels for a batch of images given text prompts.

        Args:
            images: Batch of images [batch_size, 3, 224, 224]
            texts: List of text prompts [num_classes]

        Returns:
            predictions: Class predictions [batch_size]
            probabilities: Class probabilities [batch_size, num_classes]
        """
        # TODO: Compute similarities: [batch_size, num_classes]
        logits = self.compute_similarity(images, texts)

        # TODO: Get predicted class indices and probabilities
        probabilities = F.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        return predictions, probabilities
