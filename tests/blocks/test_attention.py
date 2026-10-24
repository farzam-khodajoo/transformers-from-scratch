"""Unit tests for src/blocks/attention.py."""
import math

import pytest
import torch

from blocks.attention import MultiHeadSelfAttention, casual_mask

# Every axis gets a different size, so a swapped dimension can't pass by accident.
SHAPES = [
    # (batch, seq_len, d_model, n_head)
    (2, 5, 16, 4),
    (3, 7, 24, 3),
    (1, 1, 8, 2),
]


def make_attention(d_model: int = 16, n_head: int = 4, dropout: float = 0.0) -> MultiHeadSelfAttention:
    return MultiHeadSelfAttention(d_model=d_model, n_head=n_head, dropout=dropout).eval()


def reference_attention(module: MultiHeadSelfAttention, x: torch.Tensor) -> torch.Tensor:
    """Causal multi-head attention written out step by step, using the module's own weights."""
    B, T, C = x.shape
    H, D = module.n_head, module.d_head
    q, k, v = module.qkv(x).split(C, dim=-1)
    q, k, v = (t.view(B, T, H, D).transpose(1, 2) for t in (q, k, v))  # (B, H, T, D)
    scores = q @ k.transpose(-2, -1) / math.sqrt(D)                       # (B, H, T, T)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    out = scores.softmax(dim=-1) @ v                                      # (B, H, T, D)
    return module.projection(out.transpose(1, 2).reshape(B, T, C))


class TestCausalMask:

    def test_shape_broadcasts_over_batch_and_heads(self):
        assert casual_mask(5).shape == (1, 1, 5, 5)

    def test_is_boolean(self):
        assert casual_mask(3).dtype == torch.bool

    def test_blocks_only_future_positions(self):
        expected = torch.tensor([
            [False, True,  True,  True],
            [False, False, True,  True],
            [False, False, False, True],
            [False, False, False, False],
        ])
        assert torch.equal(casual_mask(4)[0, 0], expected)

    def test_created_on_requested_device(self):
        assert casual_mask(3, device="cpu").device.type == "cpu"


class TestConstruction:

    def test_rejects_d_model_not_divisible_by_n_head(self):
        with pytest.raises(AssertionError):
            MultiHeadSelfAttention(d_model=10, n_head=3)

    def test_head_dimension(self):
        attn = MultiHeadSelfAttention(d_model=24, n_head=3)
        assert attn.d_head == 8

    def test_parameter_shapes(self):
        attn = MultiHeadSelfAttention(d_model=16, n_head=4)
        assert attn.qkv.weight.shape == (3 * 16, 16)
        assert attn.projection.weight.shape == (16, 16)


class TestShapes:

    @pytest.mark.parametrize("batch, seq_len, d_model, n_head", SHAPES)
    def test_output_shape_matches_input(self, batch, seq_len, d_model, n_head):
        attn = make_attention(d_model, n_head)
        x = torch.randn(batch, seq_len, d_model)
        assert attn(x).shape == (batch, seq_len, d_model)

    def test_rejects_non_3d_input(self):
        attn = make_attention()
        with pytest.raises(AssertionError):
            attn(torch.randn(5, 16))

    def test_rejects_wrong_embedding_size(self):
        attn = make_attention(d_model=16)
        with pytest.raises((AssertionError, RuntimeError)):
            attn(torch.randn(2, 5, 12))


class TestBehaviour:

    def test_is_causal(self):
        """Changing the last token must not change the output at any earlier position."""
        attn = make_attention()
        x = torch.randn(2, 6, 16)
        x_changed = x.clone()
        x_changed[:, -1] += 10.0
        with torch.no_grad():
            out, out_changed = attn(x), attn(x_changed)
        assert torch.allclose(out[:, :-1], out_changed[:, :-1], atol=1e-6)

    @pytest.mark.parametrize("batch, seq_len, d_model, n_head", SHAPES)
    def test_matches_reference_implementation(self, batch, seq_len, d_model, n_head):
        attn = make_attention(d_model, n_head)
        x = torch.randn(batch, seq_len, d_model)
        with torch.no_grad():
            assert torch.allclose(attn(x), reference_attention(attn, x), atol=1e-5)

    def test_batch_items_are_independent(self):
        """Changing one sequence in the batch must not affect the others."""
        attn = make_attention()
        x = torch.randn(3, 5, 16)
        x_changed = x.clone()
        x_changed[1] = torch.randn(5, 16)
        with torch.no_grad():
            out, out_changed = attn(x), attn(x_changed)
        assert torch.allclose(out[0], out_changed[0], atol=1e-6)
        assert torch.allclose(out[2], out_changed[2], atol=1e-6)

    def test_eval_mode_is_deterministic(self):
        """In eval mode dropout must be off: the same input gives the same output."""
        attn = make_attention(dropout=0.5)
        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            assert torch.equal(attn(x), attn(x))

    def test_train_mode_applies_dropout(self):
        attn = make_attention(dropout=0.5).train()
        x = torch.randn(2, 5, 16)
        assert not torch.equal(attn(x), attn(x))

    def test_gradients_reach_all_parameters(self):
        attn = make_attention().train()
        attn(torch.randn(2, 5, 16)).sum().backward()
        for name, param in attn.named_parameters():
            assert param.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
            assert param.grad.abs().sum() > 0, f"zero gradient for {name}"

    def test_supports_float64(self):
        attn = make_attention().double()
        out = attn(torch.randn(2, 5, 16, dtype=torch.float64))
        assert out.dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_runs_on_gpu(self):
        attn = make_attention().cuda()
        out = attn(torch.randn(2, 5, 16, device="cuda"))
        assert out.device.type == "cuda"
