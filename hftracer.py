from contextlib import contextmanager
import time
import math
import torch
from transformers import AutoTokenizer, AutoModel


@contextmanager
def trace(name, **meta):
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        meta_str = " ".join(f"{k}={v}" for k, v in meta.items())
        print(f"[trace] {name} {dt:.3f}s {meta_str}".rstrip())


@contextmanager
def span(name, **meta):
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        meta_str = " ".join(f"{k}={v}" for k, v in meta.items())
        print(f"  [span] {name} {dt:.3f}s {meta_str}".rstrip())


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def l2_normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


affiliation = """[Amthor, H.] Inst Myol, UPMC INSERM, UMR S 974, CNRS,UMR 7215, F-75013 Paris, France
University of London; King's College London; University of London; University College London; Saitama Medical University; Sorbonne Universite; Institut National de la Sante et de la Recherche Medicale (Inserm); Centre National de la Recherche Scientifique (CNRS)"""

topics = [
    "Academic (universities, colleges)",
    "Government (ministries, agencies, national labs)",
    "Private (for-profit)",
    "Nonprofit/NGO",
    "Intergovernmental/Multilateral",
]

model_name = "intfloat/e5-base-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

with trace(
    "hf-embedding-similarity",
    model=model_name,
    device=device,
    n_texts=1 + len(topics),
):
    with span("load-model"):
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

    texts = [affiliation] + topics
    with span("tokenize", n_inputs=len(texts)):
        batch = tok(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

    with torch.inference_mode():
        with span("forward"):
            out = model(**batch)
            emb = mean_pool(out.last_hidden_state, batch["attention_mask"])
            emb = l2_normalize(emb).cpu()

    a = emb[0]
    for topic, b in zip(topics, emb[1:]):
        s = float((a * b).sum())
        print(f"{s:.6f}\t{topic}")
