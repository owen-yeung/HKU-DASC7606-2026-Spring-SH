"""
Text encoder for CLIP model using RoBERTa.
"""

import torch
import torch.nn as nn

from transformers import RobertaConfig, RobertaModel, RobertaTokenizer


class TextEncoder(nn.Module):
    """
    Text encoder for CLIP model using RoBERTa.
    Encodes text descriptions and projects them to the embedding space.
    """

    def __init__(self, embed_dim=512, pretrained=True, trainable_text_layers=0):
        super().__init__()
        # Load pretrained RoBERTa model and tokenizer
        if pretrained:
            self.roberta = RobertaModel.from_pretrained(
                "roberta-base", add_pooling_layer=False
            )
        else:
            self.roberta = RobertaModel(
                RobertaConfig.from_pretrained("roberta-base"), add_pooling_layer=False
            )
        self.feature_dim = self.roberta.config.hidden_size

        # Freeze pretrained RoBERTa parameters by default.
        for param in self.roberta.parameters():
            param.requires_grad = False

        # Unfreeze only the last transformer layers for parameter-efficient tuning.
        if trainable_text_layers > 0:
            layers = self.roberta.encoder.layer
            for layer in layers[-trainable_text_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # TODO: Create self.projection: a Sequential network that maps RoBERTa features to embed_dim
        # Architecture: Linear -> ReLU -> Linear
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, embed_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode tokenized text into embeddings.
        Args:
            input_ids: Tokenized text [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
        Returns:
            embeddings: Text embeddings [batch_size, embed_dim]
        """
        # TODO: Get RoBERTa outputs
        # Shape of last_hidden_state: [batch_size, seq_length, hidden_dim]
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # TODO: Extract token-level embeddings
        token_embeddings = outputs.last_hidden_state

        # TODO: Perform mean pooling over non-padding tokens using attention_mask
        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
        masked_embeddings = token_embeddings * mask
        summed = masked_embeddings.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-9)
        sentence_embeddings = summed / lengths

        # TODO: Project sentence embeddings to embed_dim
        embeddings = self.projection(sentence_embeddings)
        return embeddings


class TextTokenizer:
    """
    Utility class for tokenizing text descriptions using RoBERTa tokenizer.
    """

    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def __call__(self, texts, device="cuda") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize text descriptions.

        Args:
            texts: List of text strings
            device: Device to put tensors on

        Returns:
            input_ids: Tokenized text
            attention_mask: Attention mask
        """
        # https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend.__call__
        encoded = self.tokenizer(texts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        return input_ids, attention_mask
