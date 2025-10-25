import torch


def make_causal_mask(L, device="cpu"):
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)


def test_causal_mask_blocks_future():
    L = 4
    mask = make_causal_mask(L)
    # verificar la diagonal del tensor , la diagonal superior
    expected = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)
