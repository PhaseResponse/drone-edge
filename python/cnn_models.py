# cnn models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import time

class SimpleCNN(nn.Module):
    """Binary classifier using standard convolutions."""
    def __init__(self, input_shape):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            n = self.net(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.fc(self.net(x)).squeeze(1)


class DepthwiseSeparableCNN(nn.Module):
    """Binary classifier using depthwise separable convolutions."""    
    def __init__(self, input_shape):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, groups=1), nn.Conv2d(1, 16, kernel_size=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16), nn.Conv2d(16, 32, kernel_size=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32), nn.Conv2d(32, 64, kernel_size=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            n = self.net(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.fc(self.net(x)).squeeze(1)        


def quantize_model(model):
    """Quantize model weights to int8 for reduced memory."""    
    torch.backends.quantized.engine = 'qnnpack'
    return torch.quantization.quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)


class SpectrogramDataset(Dataset):
    """PyTorch Dataset for spectrogram inputs with binary labels."""    
    def __init__(self, segments):
        self.segments = segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        spec, label = self.segments[idx]
        x = torch.tensor(spec).unsqueeze(0)  # (1, freq, time)
        y = torch.tensor(label, dtype=torch.float32)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        return x, y    


def train(model, loader, epochs=10, lr=1e-3, start_epoch=0, pos_weight=torch.tensor([1.0])): 
    """Train model. After each epoch, saves model for inference and checkpoint for continued training."""    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"using device: {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    print(f'pos_weight={pos_weight.item():.2f}')
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.4f} time={time.time()-t0:.1f}s')
        torch.save(model.state_dict(), 'models/best_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': total_loss
        }, 'models/checkpoint.pt')
        
    return model

    