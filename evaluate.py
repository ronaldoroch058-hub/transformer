"""Evaluate the trained Transformer on the held-out validation split.

Translates every validation sentence with greedy decoding and reports
corpus BLEU / chrF (sacreBLEU) plus character and word error rates.
Results are written to the results/ folder.

Usage:
    python evaluate.py              # latest checkpoint, full validation set
    python evaluate.py --limit 200  # quick run on the first 200 sentences
"""
import argparse
import warnings
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF
import torch
import torchmetrics
from tqdm import tqdm

from config import get_config, latest_weights_file_path
from train import get_ds, get_model, greedy_decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only evaluate the first N validation sentences")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = get_config()
    device = torch.device(args.device)

    _, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    val_ds = val_dataloader.dataset

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    weights = latest_weights_file_path(config)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"Loaded {weights} (epoch {state['epoch']})")

    n = len(val_ds) if args.limit is None else min(args.limit, len(val_ds))
    sources, references, predictions = [], [], []
    with torch.no_grad():
        for idx in tqdm(range(n), desc="Translating"):
            item = val_ds[idx]
            encoder_input = item["encoder_input"].unsqueeze(0).to(device)
            encoder_mask = item["encoder_mask"].unsqueeze(0).to(device)
            out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config["seq_len"], device)
            sources.append(item["src_text"])
            references.append(item["tgt_text"])
            predictions.append(tokenizer_tgt.decode(out.detach().cpu().numpy()))

    bleu_metric, chrf_metric = BLEU(), CHRF()
    bleu = bleu_metric.corpus_score(predictions, [references])
    chrf = chrf_metric.corpus_score(predictions, [references])
    cer = torchmetrics.text.CharErrorRate()(predictions, references).item()
    wer = torchmetrics.text.WordErrorRate()(predictions, references).item()

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    summary = (
        f"checkpoint: {weights} (epoch {state['epoch']})\n"
        f"sentences: {n}\n"
        f"BLEU: {bleu.score:.2f}\n"
        f"chrF: {chrf.score:.2f}\n"
        f"CER: {cer:.4f}\n"
        f"WER: {wer:.4f}\n"
        f"sacrebleu: {bleu} | {bleu_metric.get_signature()}\n"
    )
    (out_dir / "metrics.txt").write_text(summary)
    with open(out_dir / "translations.tsv", "w") as f:
        f.write("source_en\treference_it\tpredicted_it\n")
        for s, r, p in zip(sources, references, predictions):
            f.write(f"{s}\t{r}\t{p}\n".replace("\r", " "))
    print(summary)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
