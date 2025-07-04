import os
import random
import numpy as np
import torch

def setup_environment():
    """Mitigate threading issues with BLAS libraries."""
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Disable parallel downloads from HuggingFace Hub to prevent "can't start new thread" errors
    # in resource-constrained environments (like some cloud instances / containers).
    os.environ['HF_HUB_DISABLE_PARALLEL_DOWNLOAD'] = '1'

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 