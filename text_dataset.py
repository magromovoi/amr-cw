import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts_df, tokenizer):
        self.data = texts_df
        self.tokenizer = tokenizer
        self.num_classes = texts_df['target'].nunique()

    def __getitem__(self, idx):
        text = str(self.data.text.iloc[idx])
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=512,
                                        return_token_type_ids=True)

        input_ids = tokenized_text['input_ids']
        attention_mask = tokenized_text['attention_mask']

        return {'custom_index': self.data.custom_index.iloc[idx],
                'label': self.data.label.iloc[idx],
                'split': self.data.split.iloc[idx],
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'target': torch.tensor(self.data.target.iloc[idx], dtype=torch.long)}

    def __len__(self):
        return len(self.data)
