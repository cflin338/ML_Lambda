from torch.utils.data import Dataset
import torch

class ReconDataset(Dataset):
    def __init__(self, recon, recon_next, curr_lam):

        # convert ONCE
        self.recon = torch.tensor(recon, dtype=torch.float32).unsqueeze(1)
        self.recon_next = torch.tensor(recon_next, dtype=torch.float32).unsqueeze(1)
        self.curr_lam = torch.tensor(curr_lam, dtype=torch.float32)

    def __len__(self):
        return self.recon.shape[0]

    def __getitem__(self, idx):
        return (
            self.recon[idx],
            self.recon_next[idx],
            self.curr_lam[idx]
        )