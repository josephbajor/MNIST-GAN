from tokenize import Double
import numpy as np
import torch

def onehot_convert(lab, num_labels):
    n = np.zeros(num_labels, dtype=np.float32)
    n[lab] = 1
    return torch.tensor(n, dtype=torch.double)


def rand_label(num_labels, onehot=True):
    lab = np.random.choice(range(num_labels),1)
    if onehot:
        return onehot_convert(lab, num_labels)
    else:
        return torch.tensor(lab, dtype=torch.double)


def rand_label_batched(hparams, num_labels:int = 10) -> torch.Tensor:
    n = np.zeros((hparams.batch_size, num_labels), dtype=np.float32)
    for i in n:
        lab = np.random.choice(range(num_labels),1)
        i[lab] = 1
    return torch.tensor(n, dtype=torch.double)
