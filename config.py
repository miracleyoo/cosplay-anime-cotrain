# coding: utf-8
import torch
import torch.nn as nn
import torchvision.models as models

class Config(object):
    def __init__(self):
        self.USE_CUDA            = torch.cuda.is_available()
        self.NET_SAVE_PATH       = "./source/trained_net/"
        self.DATASET_PATH        = "cosplay"
        self.TRAINDATARATIO      = 0.7
        self.RE_TRAIN            = False
        self.PIC_SIZE            = 256
        self.NUM_TEST            = 0
        self.NUM_TRAIN           = 0
        self.TOP_NUM             = 1
        self.NUM_EPOCHS          = 50
        self.BATCH_SIZE          = 1
        self.TEST_BATCH_SIZE     = 8
        self.NUM_WORKERS         = 4
        self.NUM_CLASSES         = 8
        self.LEARNING_RATE       = 0.001
        self.PRINT_BATCH         = True
        self.NUM_PRINT_BATCH     = 5

