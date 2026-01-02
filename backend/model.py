# tiny diffusion - char level text gen
# based on nathan-barry/tiny-diffusion, modified for api use
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Generator, Callable, Optional
import threading, os

N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE = 384, 6, 6, 256

def norm(x): return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1*cos + x2*sin, x1*(-sin) + x2*cos], 3).to(x.dtype)

class MultiHeadAttention(nn.Module):
    def __init__(s, n_embd, n_head):
        super().__init__()
        s.n_head, s.head_dim = n_head, n_embd // n_head
        s.c_q = nn.Linear(n_embd, n_embd, bias=False)
        s.c_k = nn.Linear(n_embd, n_embd, bias=False)
        s.c_v = nn.Linear(n_embd, n_embd, bias=False)
        s.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(s, x, cos_sin):
        B, T, C = x.size()
        q = s.c_q(x).view(B, T, s.n_head, s.head_dim)
        k = s.c_k(x).view(B, T, s.n_head, s.head_dim)
        v = s.c_v(x).view(B, T, s.n_head, s.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # qk norm helps stability
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return s.c_proj(y.transpose(1,2).contiguous().view(B, T, -1))

class MLP(nn.Module):
    def __init__(s, n_embd):
        super().__init__()
        s.c_fc = nn.Linear(n_embd, 4*n_embd, bias=False)
        s.c_proj = nn.Linear(4*n_embd, n_embd, bias=False)
    def forward(s, x): return s.c_proj(F.relu(s.c_fc(x)).square())  # squared relu

class Block(nn.Module):
    def __init__(s, n_embd, n_head):
        super().__init__()
        s.attn = MultiHeadAttention(n_embd, n_head)
        s.mlp = MLP(n_embd)
    def forward(s, x, cos_sin):
        x = x + s.attn(norm(x), cos_sin)
        return x + s.mlp(norm(x))

class DiffusionModel(nn.Module):
    def __init__(s, vocab_size, n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER, block_size=BLOCK_SIZE):
        super().__init__()
        s.vocab_size = vocab_size
        s.block_size = block_size
        s.head_dim = n_embd // n_head
        s.token_emb = nn.Embedding(vocab_size, n_embd)
        s.rotary_seq_len = block_size * 2
        cos, sin = s._precompute_rotary(s.rotary_seq_len)
        s.register_buffer("cos", cos, persistent=False)
        s.register_buffer("sin", sin, persistent=False)
        s.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        s.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        s.apply(s._init_weights)

    def _init_weights(s, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding): torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _precompute_rotary(s, seq_len, base=10000):
        inv_freq = 1.0 / (base ** (torch.arange(0, s.head_dim, 2, dtype=torch.float32) / s.head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[None, :, None, :], sin[None, :, None, :]

    def forward(s, idx, targets=None, mask=None):
        B, T = idx.size()
        x = norm(s.token_emb(idx))
        cos_sin = (s.cos[:, :T], s.sin[:, :T])
        for block in s.blocks: x = block(x, cos_sin)
        logits = s.lm_head(norm(x))
        if targets is None: return logits, None
        if mask is not None:
            loss = F.cross_entropy(logits.view(-1, s.vocab_size), targets.view(-1), reduction="none")
            return logits, (loss * mask.view(-1)).sum() / mask.view(-1).sum()
        return logits, F.cross_entropy(logits.view(-1, s.vocab_size), targets.view(-1))


class TinyDiffusionModel:
    """wrapper for inference, handles loading + generation"""
    def __init__(s):
        s.model = None
        s.device = None
        s.loading = False
        s.loaded = False
        s.load_progress = 0
        s.load_error = None
        s._lock = threading.Lock()
        # vocab stuff
        s.chars = None
        s.stoi = None
        s.itos = None
        s.mask_token_id = 0
        s.vocab_size = 66  # tiny shakespeare + mask token
        s.block_size = BLOCK_SIZE
        s.data = None

    def get_device(s):
        if torch.backends.mps.is_available():
            try: 
                torch.zeros(1, device="mps")  # test if mps actually works
                return "mps"
            except: pass
        return "cpu"

    def encode(s, txt): return [s.stoi.get(c, s.mask_token_id) for c in txt]
    def decode(s, l): return ''.join([s.itos.get(i, '_') for i in l])

    def load_model(s, progress_callback=None):
        with s._lock:
            if s.loaded or s.loading: return
            s.loading = True
            s.load_error = None
        try:
            if progress_callback: progress_callback(10, "Loading vocab...")
            
            # load training data for vocab
            data_path = os.path.join(os.path.dirname(__file__), '..', 'tiny_diffusion_model', 'data.txt')
            with open(data_path, 'r') as f: text = f.read()
            
            # build vocab - underscore is mask token at idx 0
            s.chars = ['_'] + sorted(list(set(text)))
            s.vocab_size = len(s.chars)
            s.stoi = {c: i for i, c in enumerate(s.chars)}
            s.itos = {i: c for i, c in enumerate(s.chars)}
            s.mask_token_id = 0
            s.data = torch.tensor(s.encode(text), dtype=torch.long)
            
            s.device = s.get_device()
            dev_name = "Apple Metal (MPS)" if s.device == "mps" else "CPU"
            if progress_callback: progress_callback(40, f"Building model on {dev_name}...")
            
            s.model = DiffusionModel(s.vocab_size)
            
            if progress_callback: progress_callback(60, "Loading weights...")
            weights_path = os.path.join(os.path.dirname(__file__), '..', 'tiny_diffusion_model', 'weights', 'diffusion.pt')
            state = torch.load(weights_path, map_location=s.device, weights_only=True)
            s.model.load_state_dict(state)
            s.model.to(s.device)
            s.model.eval()
            
            if progress_callback: progress_callback(100, f"Dlaude Ready ({dev_name})")
            with s._lock: s.loaded, s.loading = True, False
        except Exception as e:
            with s._lock: s.loading, s.load_error = False, str(e)
            raise

    @torch.no_grad()
    def generate(s, prompt: str, max_tokens: int = 512, temperature: float = 0.8, 
                 steps: int = 32, block_length: int = 32, step_callback=None) -> Generator[str, None, None]:
        """diffusion generation - iteratively unmask tokens based on confidence"""
        if not s.loaded: raise RuntimeError("Model not loaded")
        
        # TODO: actually use the prompt instead of hardcoded shakespeare start
        # for now just use first 16 chars of training data as seed
        prompt_len = 16
        all_tokens = s.data[:prompt_len].tolist()
        total_steps = 0
        
        if step_callback: step_callback(0, steps, s.decode(all_tokens))
        
        block_len = min(240, max_tokens)
        
        # init sequence: prompt + masked positions
        x = torch.full((1, s.block_size), s.mask_token_id, dtype=torch.long, device=s.device)
        x[0, :prompt_len] = torch.tensor(all_tokens, device=s.device)
        masked = torch.zeros(1, s.block_size, dtype=torch.bool, device=s.device)
        masked[0, prompt_len:prompt_len + block_len] = True
        
        # iterative unmasking
        while masked.any():
            total_steps += 1
            logits, _ = s.model(x)
            probs = F.softmax(logits / temperature, dim=-1)
            
            # confidence = sum of top-3 probs
            top_k = 3
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            confidences = top_k_probs.sum(dim=-1)
            
            # unmask high confidence positions
            decode_mask = (confidences >= 0.95) & masked
            if not decode_mask.any():
                # if nothing passes threshold, unmask the most confident one
                masked_conf = torch.where(masked, confidences, torch.tensor(-float("inf"), device=s.device))
                decode_mask.view(-1)[masked_conf.argmax()] = True
            
            # sample from top-k (avoid multinomial errors with eps)
            top_k_sum = top_k_probs.sum(dim=-1, keepdim=True)
            top_k_norm = top_k_probs / (top_k_sum + 1e-8)
            top_k_norm = torch.clamp(top_k_norm, min=1e-8, max=1.0)
            top_k_norm = top_k_norm / top_k_norm.sum(dim=-1, keepdim=True)  # renormalize just in case
            
            sampled_k = torch.multinomial(top_k_norm.view(-1, top_k), 1).view(1, s.block_size)
            sampled_tokens = torch.gather(top_k_indices, -1, sampled_k.unsqueeze(-1)).squeeze(-1)
            
            x = torch.where(decode_mask, sampled_tokens, x)
            masked = masked & ~decode_mask
            
            if step_callback and total_steps % 5 == 0:
                step_callback(min(steps, total_steps), steps, s.decode(x[0, :prompt_len + block_len].tolist()))
        
        result = s.decode(x[0, :prompt_len + block_len].tolist())
        if step_callback: step_callback(steps, steps, result)
        yield result


# singleton
_model_instance = None
def get_model():
    global _model_instance
    if _model_instance is None: _model_instance = TinyDiffusionModel()
    return _model_instance
