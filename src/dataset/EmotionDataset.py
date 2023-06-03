"""
The requirment from datasets are:
1. they should delete non important stuff
2. they should tokenize the input
"""
from torch.utils.data import DataLoader, Dataset
import torch 
from transformers import DataCollatorWithPadding

class EmotionDataset(Dataset):
    def __init__(self, df, config, has_labels=True):
        super().__init__()
        self.data_column = 'text'
        self.class_column = 'label'
        self.data = df
        self.config = config
        self.has_labels = has_labels

    def __getitem__(self, idx):
        text = self.data.loc[idx, self.data_column]
        if(self.has_labels):
            label = self.data.loc[idx, self.class_column]
        inputs = self.config.tokenizer(text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=self.config.general.max_length,
            pad_to_max_length=True,
            truncation=True,
        )

        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)

        if(self.has_labels):
            label = torch.tensor(label, dtype=torch.long)
            return inputs, label
        else:
            return inputs
    
    def __len__(self):
        return self.data.shape[0]
    


def get_train_dataloader(config, train_df):
    dataset = EmotionDataset(train_df, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.general.train_batch_size,
        num_workers=config.general.n_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader

def get_valid_dataloader(config, df):
    dataset = EmotionDataset(df, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.general.train_batch_size,
        num_workers=config.general.n_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader

def get_test_dataloader(config, df):
    dataset = EmotionDataset(df, config, has_labels=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.general.train_batch_size,
        shuffle=False,
        num_workers=config.general.n_workers,
        pin_memory=True,
        drop_last=False
    )
    return dataloader