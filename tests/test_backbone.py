from types import SimpleNamespace

import torch
from torch import nn

from embodi.backbone import LFMBackboneAdapter


class FakeLFM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(
                hidden_size=12,
                layer_types=["conv", "full_attention", "conv", "full_attention", "conv"],
            )
        )
        self.embedding = nn.Embedding(32, 12)
        self.vision_tower = nn.Linear(2, 2)
        self.received_shape = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, **kwargs):
        self.received_shape = inputs_embeds.shape
        hidden_states = tuple(inputs_embeds + index for index in range(6))
        return SimpleNamespace(hidden_states=hidden_states)


def test_adapter_appends_state_and_taps_last_attention_output() -> None:
    model = FakeLFM()
    adapter = LFMBackboneAdapter(model, state_dim=10)
    input_ids = torch.ones(2, 4, dtype=torch.long)
    features, mask = adapter(
        torch.randn(2, 4, 10),
        state_mask=torch.tensor([[True, False, False, False], [True, True, False, False]]),
        input_ids=input_ids,
        attention_mask=torch.ones(2, 4, dtype=torch.long),
    )
    assert model.received_shape == (2, 8, 12)
    assert features.shape == (2, 8, 12)
    assert mask.shape == (2, 8)
    assert mask[:, 4:].tolist() == [[True, False, False, False], [True, True, False, False]]
    # Last full_attention is layer 3, represented by hidden_states[layer + 1].
    torch.testing.assert_close(features[:, :4], model.embedding(input_ids) + 4)
