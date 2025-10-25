import argparse, tarfile, io, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from decoder import build_decoder_only


class TokenDataset(Dataset):
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                ids = obj["ids"]
                if len(ids) >= 2:
                    self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.samples[idx][1:], dtype=torch.long)
        return x, y


def collate_batch(batch):
    x_batch, y_batch = zip(*batch)
    x_batch = nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=0)
    y_batch = nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=0)
    return x_batch, y_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Dataset
    dataset = TokenDataset(args.input)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch
    )

    # Obtener vocab size
    vocab_size = sum(1 for _ in open("out/vocab.txt", encoding="utf-8"))

    # Modelo
    model = build_decoder_only(
        vocab_size=vocab_size,
        max_len=args.seq_len,
        d_model=args.dim,
        n_heads=args.heads,
        n_layers=2,
        dropout=0.1,
        tie_weights=True,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # Entrenamiento
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device)
            logits = model(x)  # (B, L, V)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1}: loss = {avg_loss:.4f}")

    # Metadatos
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)

    meta = {
        "lr": args.lr,
        "heads": args.heads,
        "dim": args.dim,
        "timestamp": int(time.time()),
        "vocab_size": vocab_size,
        "seq_len": args.seq_len,
        "notes": "Trained decoder-only model",
    }

    with tarfile.open(args.output, "w:gz") as tar:
        # Modelo
        model_info = tarfile.TarInfo("checkpoint/model.pt")
        model_info.size = buffer.getbuffer().nbytes
        tar.addfile(model_info, buffer)

        # Metadatos
        meta_bytes = json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
        meta_info = tarfile.TarInfo("checkpoint/meta.json")
        meta_info.size = len(meta_bytes)
        tar.addfile(meta_info, io.BytesIO(meta_bytes))


if __name__ == "__main__":
    main()
