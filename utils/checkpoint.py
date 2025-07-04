import torch
from config import Config
from models.rcop import RCOP

def load_model_from_checkpoint(checkpoint_path, num_speakers, num_phones, device):
    """Load RCOP model from checkpoint."""
    config = Config()
    
    # Load checkpoint first to get metadata
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to extract num_speakers and num_phones from checkpoint metadata
    if 'num_speakers' in checkpoint:
        num_speakers = checkpoint['num_speakers']
    if 'num_phones' in checkpoint:
        num_phones = checkpoint['num_phones']
    
    # Infer from model state dict if available (fallback)
    if num_speakers is None and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'sp_clf.weight' in state_dict:
            num_speakers = state_dict['sp_clf.weight'].size(0)
    
    if num_speakers is None or num_phones is None:
        raise ValueError("Could not determine num_speakers or num_phones from checkpoint.")

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

def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path, num_speakers, num_phones, best_val_loss=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'num_speakers': num_speakers,
        'num_phones': num_phones,
    }
    
    # Add metadata for proper model loading
    if best_val_loss is not None:
        checkpoint['best_val_loss'] = best_val_loss
        
    torch.save(checkpoint, save_path) 