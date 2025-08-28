"""
Sliding window perplexity calculations. 
Taken mostly from: https://huggingface.co/docs/transformers/perplexity 

"""

import numpy  as np
import pandas as pd
import math
import torch

from typing import Iterable, Tuple

from transformers import PreTrainedTokenizerBase, PreTrainedModel
#from transformers import AutoTokenizer, AutoModelForCausalLM


# ====================================================================
# Get device
# ====================================================================
def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    try: 
        if torch.backends.mps.is_available(): 
            return torch.device("mps")
    except Exception: pass
    return torch.device("cpu")


# ====================================================================
# Core scorer (sliding window, token-weighted)
# ====================================================================
@torch.inference_mode()
def perplexity_sliding(
    text        : str,
    DEVICE      : torch.device, 
    tokenizer   : PreTrainedTokenizerBase, 
    model       : PreTrainedModel, 
    stride      : int        = 256, 
    max_length  : int | None = None,
    ) -> Tuple[float, int, float]:
    """
    Returns: (perplexity, num_scored_tokens, average_negative_log_likelihood)
    - Token-weighted average over sliding windows
    - If the sequence is short, it just does one pass
    """
    if not text or text.strip() == "": return float("nan"), 0, float("nan")

    # Tokenize
    enc       = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(DEVICE)
    seq_len   = input_ids.size(1)

    # Max length with fallback
    if max_length is None: max_length = getattr(model.config, "n_positions", 1024)

    # Book-keeping
    nll_sum  = 0.0
    n_tokens = 0
    prev_end = 0

    # --------------------------------------------------------------------
    # Loop over sliding windows
    # --------------------------------------------------------------------
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end  # How many new tokens we score in this step
        if trg_len <= 0: break

        ids = input_ids[:, begin:end]
        target_ids = ids.clone()
        
        # Mask out the context tokens in this window
        target_ids[:, :-trg_len] = -100  # ignored by loss

        # outputs.loss is avg NLL over the valid labels in this window
        outputs   = model(ids, labels=target_ids) 
        num_valid = int((target_ids != -100).sum().item())
        
        # Model internally shifts labels left by 1, so it scores (num_valid - batch_size) tokens
        loss_tokens = max(num_valid - ids.size(0), 0)

        if loss_tokens > 0:
            nll_sum  += float(outputs.loss) * loss_tokens
            n_tokens += loss_tokens

        # Continue with the loop
        prev_end = end
        if end == seq_len: break

    # Too short to get any scored tokens (1-token strings)
    if n_tokens == 0: return float("nan"), 0, float("nan")

    # Average negative log-likelihood & perplexity
    average_negative_log_likelihood = nll_sum / n_tokens
    perplexity                      = math.exp(average_negative_log_likelihood)

    return perplexity, n_tokens, average_negative_log_likelihood


# ====================================================================
# Batch helper (list or DataFrame)
# ====================================================================
def score_utterances(
    utterances  : Iterable[str], 
    DEVICE      : torch.device, 
    tokenizer   : PreTrainedTokenizerBase, 
    model       : PreTrainedModel, 
    stride      : int = 1, 
    max_length  : int = 8,
    ):
    """
    ppl     => perplexity
    avg_nll => average negative log-likelihood
    """
    out = []
    for i, utt in enumerate(utterances):
        ppl, n_tok, avg_nll = perplexity_sliding(utt, DEVICE, tokenizer, model, stride=stride, max_length=max_length)
        out.append({"idx": i, "text": utt, "tokens_scored": n_tok, "avg_nll": avg_nll, "perplexity": ppl})
    return out


# ====================================================================
# Agggregate Scores (when there are multiple utterances)
# ====================================================================
def combine_user_group(
    g          : pd.DataFrame,
    pID        : str,
    *,
    outlier    : str                 = "winsor",      # 'none' | 'trim' | 'winsor' | 'huber' | 'median'
    trim_q     : Tuple[float, float] = (0.05, 0.95),  # used for 'trim' and default bounds for 'winsor'
    huber_k    : float               = 1.345,         # Huber tuning constant
    min_tokens : int                 = 5              # drop utterances with < this many scored tokens
):
    # --------------------------------------------------------------------
    # Helper functions
    # --------------------------------------------------------------------
    def _wquantile(x, w, q):
        idx = np.argsort(x)
        x, w = np.asarray(x)[idx], np.asarray(w)[idx]
        cw = np.cumsum(w)
        if cw[-1] == 0: return np.nan
        cw = cw / cw[-1]
        return np.interp(q, cw, x)

    def _wmedian(x, w):
        return _wquantile(x, w, 0.5)

    def _wmad(x, w):
        m = _wmedian(x, w)
        d = np.abs(np.asarray(x) - m)
        return _wquantile(d, w, 0.5)

    # --------------------------------------------------------------------
    # Initial empty/validity checks
    # --------------------------------------------------------------------
    # Empty
    g = g.copy()
    g = g[g['tokens_scored'] > 0]
    g = g[g['tokens_scored'] >= min_tokens]
    if g.empty:
        return {"pID": pID, 'tokens': 0, 'avg_nll': float('nan'), 'ppl': float('nan'), 'bpt': float('nan')}

    # Validity
    x = g['avg_nll'].to_numpy()
    w = g['tokens_scored'].to_numpy()
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[valid], w[valid]
    if x.size == 0:
        return {"pID": pID, 'tokens': 0, 'avg_nll': float('nan'), 'ppl': float('nan'), 'bpt': float('nan')}

    # Keep a token count that's independent of reweighting
    tokens_kept = int(w.sum())

    # --------------------------------------------------------------------
    # Decide how to handle outliers
    # --------------------------------------------------------------------
    if outlier == "trim":
        lo, hi = _wquantile(x, w, trim_q[0]), _wquantile(x, w, trim_q[1])
        keep = (x >= lo) & (x <= hi)
        x, w = x[keep], w[keep]
        tokens_kept = int(w.sum())  # after trimming
        
    elif outlier == "winsor":
        lo, hi = _wquantile(x, w, trim_q[0]), _wquantile(x, w, trim_q[1])
        x = np.clip(x, lo, hi)
        # tokens_kept unchanged
        
    elif outlier == "huber":
        m = _wmedian(x, w)
        mad = _wmad(x, w)
        s = max(1e-6, mad * 1.4826)     # robust scale estimate
        r = (x - m) / s
        # Huber reweighting (soft down-weight outliers)
        w = w * np.minimum(1.0, huber_k / np.maximum(np.abs(r), 1e-12))
        
    elif outlier == "median":
        avg_nll = _wmedian(x, w)
        ppl = math.exp(avg_nll)
        bpt = avg_nll / math.log(2)
        return {"pID": pID, 'tokens': tokens_kept, 'avg_nll': avg_nll, 'ppl': ppl, 'bpt': bpt}
    # else 'none' -> do nothing

    # Again check to make sure we still have tokens
    if w.sum() == 0:
        return {"pID": pID, 'tokens': 0, 'avg_nll': float('nan'), 'ppl': float('nan'), 'bpt': float('nan')}

    # --------------------------------------------------------------------
    # Final recalculation of NLL and Perplexity
    # --------------------------------------------------------------------
    avg_nll = float(np.average(x, weights=w))
    ppl     = math.exp(avg_nll)
    bpt     = avg_nll / math.log(2)

    return {"pID": pID, 'tokens': tokens_kept, 'avg_nll': avg_nll, 'ppl': ppl, 'bpt': bpt}

