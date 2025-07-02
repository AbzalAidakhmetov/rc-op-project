import torch
from config import Config
from models.rcop import RCOP

def load_model_from_checkpoint(checkpoint_path, num_speakers, num_phones, device):
    """Load RCOP model from checkpoint."""
    config = Config()
    
    # Load checkpoint first to get metadata
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to extract num_speakers from checkpoint metadata
    if 'num_speakers' in checkpoint:
        num_speakers = checkpoint['num_speakers']
    elif 'model_state_dict' in checkpoint:
        # Infer from model state dict if available
        state_dict = checkpoint['model_state_dict']
        if 'sp_clf.weight' in state_dict:
            num_speakers = state_dict['sp_clf.weight'].size(0)
    
    # Initialize model
    model = RCOP(
        d_spk=config.d_spk,
        d_ssl=config.d_ssl,
        n_phones=num_phones,
        n_spk=num_speakers
    )
    
    # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, num_speakers

def save_checkpoint(model, optimizer, epoch, loss, save_path, num_speakers=None, num_phones=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Add metadata for proper model loading
    if num_speakers is not None:
        checkpoint['num_speakers'] = num_speakers
    if num_phones is not None:
        checkpoint['num_phones'] = num_phones
        
    torch.save(checkpoint, save_path) 