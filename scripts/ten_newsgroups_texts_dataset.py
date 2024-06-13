import torch
from torch.utils.data import Dataset


class TenNewsGroupsTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = dataframe['target'].nunique()

    def __getitem__(self, idx):
        text = str(self.data.text.iloc[idx])
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length,
                                        return_token_type_ids=True)

        input_ids = tokenized_text['input_ids']
        attention_mask = tokenized_text['attention_mask']

        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'targets': torch.tensor(self.data.target.iloc[idx], dtype=torch.long)}

    def __len__(self):
        return len(self.data)
