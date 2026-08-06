from __future__ import annotations

import torch
from torch import nn


class DINOEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze: bool = True,
        image_size: int = 224,
        raw_num_tokens: int | None = 196,
        num_register_tokens: int | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name)
        self.image_size = image_size
        self.raw_num_tokens = raw_num_tokens
        self.num_register_tokens = num_register_tokens
        self.freeze = freeze
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    @staticmethod
    def _select_patch_tokens(
        hidden: torch.Tensor,
        raw_num_tokens: int | None,
        num_register_tokens: int | None,
    ) -> torch.Tensor:
        if num_register_tokens is not None:
            offset = 1 + int(num_register_tokens)
            patch_tokens = hidden[:, offset:, :]
            if raw_num_tokens is not None:
                patch_tokens = patch_tokens[:, : int(raw_num_tokens), :]
        elif raw_num_tokens is not None and hidden.shape[1] >= int(raw_num_tokens):
            patch_tokens = hidden[:, -int(raw_num_tokens) :, :]
        else:
            patch_tokens = hidden[:, 1:, :]
        if raw_num_tokens is not None and patch_tokens.shape[1] != int(raw_num_tokens):
            raise ValueError(
                f"Expected {raw_num_tokens} patch tokens, got {patch_tokens.shape[1]} "
                f"from hidden sequence length {hidden.shape[1]}."
            )
        return patch_tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode normalized images.

        Args:
            images: [B, 3, H, W]
        Returns:
            patch_tokens: [B, N, D]
        """
        if self.freeze:
            with torch.no_grad():
                outputs = self.model(pixel_values=images)
                hidden = outputs.last_hidden_state
                return self._select_patch_tokens(hidden, self.raw_num_tokens, self.num_register_tokens)
        outputs = self.model(pixel_values=images)
        hidden = outputs.last_hidden_state
        return self._select_patch_tokens(hidden, self.raw_num_tokens, self.num_register_tokens)
