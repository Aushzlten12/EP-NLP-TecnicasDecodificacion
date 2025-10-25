import torch
import transformer
from torch import nn
import torch.nn.functional as F


# Técnicas de decodificación
@torch.no_grad()
def generate(
    self,
    idx,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None,
    greedy=False,
    return_probs=False,
):
    collected_probs = []
    for _ in range(max_new_tokens):
        logits = self(idx)
        logits = logits[:, -1, :] / temperature  # temperatura aplicada

        probs = F.softmax(logits, dim=-1)
        if return_probs:
            collected_probs.append(probs[0].cpu())

        if greedy:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            if top_k is not None:
                top_k_probs, top_k_indices = torch.topk(probs, top_k)
                next_token = top_k_indices.gather(-1, torch.multinomial(top_k_probs, 1))
            elif top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum_probs = sorted_probs.cumsum(dim=-1)
                mask = cum_probs > top_p
                mask[..., 0] = False
                sorted_probs[mask] = 0.0
                total = sorted_probs.sum(dim=-1, keepdim=True)
                if (total == 0).any():
                    next_token = sorted_indices[:, :1]
                else:
                    sorted_probs = sorted_probs / total
                    next_token = sorted_indices.gather(
                        -1, torch.multinomial(sorted_probs, 1)
                    )
            else:
                next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

    if return_probs:
        return idx, collected_probs
    return idx


@torch.no_grad()
def beam_search(model, idx, max_new_tokens, beam_width=3, length_penalty=1.0):
    sequences = [(idx, 0)]  # Cada item (tokens, log_prob_acumulado)

    for _ in range(max_new_tokens):
        all_candidates = []
        for seq, score in sequences:
            logits = model(seq)
            logits = logits[:, -1, :]
            probs = F.log_softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, beam_width)

            for i in range(beam_width):
                next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, next_token], dim=1)
                # Penalización por longitud evita que beams de respuestas cortas
                new_score = score + topk_probs[0, i].item() / (
                    len(new_seq) ** length_penalty
                )
                all_candidates.append((new_seq, new_score))

        # Ordenar y mantener los mejores
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[
            :beam_width
        ]

    return sequences[0][0]


def causal_mask(L: int, device=None):
    return torch.triu(
        torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
    ).view(1, 1, L, L)


class DecoderOnlyBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = transformer.MultiHeadAttentionBlock(d_model, n_heads, dropout)
        self.ff = transformer.FeedForwardBlock(d_model, d_ff, dropout)
        self.res1 = transformer.ResidualConnection(d_model, dropout)
        self.res2 = transformer.ResidualConnection(d_model, dropout)

    def forward(self, x, attn_mask):
        x = self.res1(x, lambda x: self.attn(x, x, x, attn_mask))
        x = self.res2(x, self.ff)
        return x


class DecoderOnlyLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        max_len=512,
        n_layers=1,
        tie_weights=True,
    ):
        super().__init__()
        self.tok = transformer.InputEmbedding(d_model, vocab_size)
        self.pos = transformer.PositionalEncoding(d_model, max_len, dropout)
        self.blocks = nn.ModuleList(
            [DecoderOnlyBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = transformer.LayerNormalization(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok.embedding.weight

    def forward(self, idx):  # (batch,seq_len)
        B, L = idx.size()
        x = self.pos(self.tok(idx))
        mask = causal_mask(L, idx.device)  # (1,1,seq_len,seq_len)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x)
        logits = self.lm_head(x)  # (Batch,seq_len,vocab_size)
        return logits


def build_decoder_only(
    vocab_size: int,
    max_len: int,
    d_model: int = 512,
    n_layers: int = 1,
    n_heads: int = 8,
    d_ff: int = 2048,
    dropout: float = 0.1,
    tie_weights: bool = True,
) -> nn.Module:
    model = DecoderOnlyLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len,
        n_layers=n_layers,
        tie_weights=tie_weights,
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
