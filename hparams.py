from dataclasses import dataclass
import torch

@dataclass
class Hparams:
    ### System ###
    data_path:str = r'D:/Repos/practice/MNIST CNN classifier/'
    model_path:str = r'D:/Repos/practice/MNIST-GAN/modelparams'
    device:str = 'gpu'
    num_workers:int = 1

    ### Model Parameters ###
    z_dim:int = 32
    kernel_size:int = 5
    batch_size:int = 512
    learn_rate:float = 4e-4

    ### Training ###
    epochs:int = 3
