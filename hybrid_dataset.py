import torch
from torch.utils.data import Dataset


class HybridDataset(Dataset):
    def __init__(self, hybrids_df):
        self.data = hybrids_df
        self.num_classes = hybrids_df['target'].nunique()
        self.embedding_size = len(hybrids_df.hybrid_embeddings.iloc[0])

    def __getitem__(self, idx):
        hybrid_embedding = self.data.hybrid_embeddings.iloc[idx]

        return {'hybrid_embedding': torch.tensor(hybrid_embedding, dtype=torch.float32),
                'target': torch.tensor(self.data.target.iloc[idx], dtype=torch.long)}

    def __len__(self):
        return len(self.data)
