from types import SimpleNamespace

import torch

from humor_generator_v35.qwen_backend import QwenBackend


class FakeMultimodalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)

    def get_input_embeddings(self):
        return self.embedding

    def get_image_features(self, pixel_values, image_grid_thw):
        del pixel_values, image_grid_thw
        return SimpleNamespace(pooler_output=(torch.full((1, 4), 17.0),))

    def get_placeholder_mask(
        self, input_ids, inputs_embeds, image_features=None, video_features=None,
    ):
        del inputs_embeds, image_features, video_features
        image = (input_ids == 9).unsqueeze(-1)
        return image, torch.zeros_like(image)

    def get_rope_index(self, input_ids, **kwargs):
        del kwargs
        positions = torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
        return positions, torch.zeros((1, 1), dtype=torch.long)


def test_backend_materializes_image_features_before_latent_insertion() -> None:
    backend = QwenBackend(FakeMultimodalModel(), processor=None, process_vision_info=None)
    inputs = {
        "input_ids": torch.tensor([[1, 9, 2]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
        "pixel_values": torch.ones((1, 3)),
        "image_grid_thw": torch.tensor([[1, 1, 1]]),
    }
    embeddings, positions = backend.multimodal_embeddings_and_positions(inputs)
    torch.testing.assert_close(embeddings[0, 1], torch.full((4,), 17.0))
    assert positions.shape == (3, 1, 3)
    assert not embeddings.requires_grad
