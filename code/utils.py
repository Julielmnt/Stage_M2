#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Functions from DeepInv"
# cff-version: 0.0.1
# authors:
# - family-names: "Tachella"
#   given-names: "Julian"
#   orcid: "https://orcid.org/0000-0003-3878-9142"
# - family-names: "Chen"
#   given-names: "Dongdong"
#   orcid: "https://orcid.org/0000-0002-7016-9288"
# - family-names: "Hurault"
#   given-names: "Samuel"
#   orcid: "https://orcid.org/0000-0002-5163-2791"
# - family-names: "Terris"
#   given-names: "Matthieu"
#   orcid: "https://orcid.org/0009-0009-9726-6131"
# title: "DeepInverse: A deep learning framework for inverse problems in imaging"
# version: latest
# doi: 10.5281/zenodo.7982256
# date-released: 2023-06-30
# url: "https://github.com/deepinv/deepinv"



import os
import torch
import numpy as np

def get_freer_gpu():
    """
    Function from Deepinv library
    Returns the GPU device with the most free memory.

    """
    try:
        if os.name == "posix":
            os.system("nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp")
            memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
        else:
            os.system('bash -c "nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp"')
            memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
        idx = np.argmax(memory_available)
        device = torch.device(f"cuda:{idx}")
        print(f"Selected GPU {idx} with {np.max(memory_available)} MB free memory ")
    except:
        device = torch.device(f"cuda")
        print("Couldn't find free GPU")

    return device