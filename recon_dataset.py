import torch
from torch.utils.data import Dataset


class ReconDataset(Dataset):
    def __init__(self, recon, residual, fft, truth):
        self.recon = recon
        self.residual = residual
        self.fft = fft
        self.truth = truth
    def __len__(self):
        return self.recon.shape[0]

    def __getitem__(self, idx):

        x_img = torch.tensor(self.recon[idx], dtype=torch.float32).unsqueeze(0)
        x_res = torch.tensor(self.residual[idx], dtype=torch.float32).unsqueeze(0)
        x_fft = torch.tensor(self.fft[idx], dtype=torch.float32).unsqueeze(0)
        x_truth = torch.from_numpy(self.truth).float().unsqueeze(0)

        return x_img, x_res, x_fft, x_truth