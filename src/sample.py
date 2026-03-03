import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_model_torch_module():
    module_path = Path(__file__).with_name('model-torch.py')
    spec = importlib.util.spec_from_file_location('model_torch', module_path)
    if spec is None or spec.loader is None:
        raise ImportError('Unable to load model-torch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model = _load_model_torch_module()

def top_k_logits(logits, k):
    if k == 0:
        return logits

    values, _ = torch.topk(logits, k=k, dim=-1)
    min_values = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < min_values, torch.finfo(logits.dtype).min)


def top_p_logits(logits, p):
    """Nucleus sampling."""
    if p >= 1:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)


def sample_sequence(
    *,
    hparams,
    length,
    start_token=None,
    batch_size=None,
    context=None,
    temperature=1,
    top_k=0,
    top_p=1,
    model_instance=None,
    device=None,
):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        if batch_size is None:
            raise ValueError('batch_size is required when start_token is specified')
        context = torch.full((batch_size, 1), start_token, dtype=torch.long)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if not torch.is_tensor(context):
        context = torch.tensor(context, dtype=torch.long)
    context = context.to(device)

    if model_instance is None:
        model_instance = model.GPT2Model(hparams)
    model_instance = model_instance.to(device)
    model_instance.eval()

    with torch.no_grad():
        output = context
        prev = context
        past = None

        for _ in range(length):
            next_outputs = model_instance(prev, past=past)
            logits = next_outputs['logits'][:, -1, :hparams.n_vocab] / float(temperature)
            logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, p=top_p)

            probs = F.softmax(logits, dim=-1)
            samples = torch.multinomial(probs, num_samples=1)

            presents = next_outputs['present']
            past = presents if past is None else torch.cat([past, presents], dim=-2)
            prev = samples
            output = torch.cat([output, samples], dim=1)

        return output
