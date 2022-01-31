import torch
import numpy as np
import sys
from hparams import Hparams
from train import train

def main():

    hparams = Hparams()

    train(hparams, load=False)


if __name__ == "__main__":
    main()
    
