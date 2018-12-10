#!/usr/bin/env python
import torch

if __name__ == '__main__':
    from feed_forward.main import run

    run(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
