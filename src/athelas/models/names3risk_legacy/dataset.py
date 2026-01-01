import torch
import torch.utils.data as data


class TextDataset(data.Dataset):

    def __init__(self, texts, tokenizer):
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode(text)
        return torch.tensor(encoding)

    def __len__(self):
        return len(self.texts)


class TabularDataset(data.Dataset):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor

    def __getitem__(self, idx):
        return self.tensor[idx]

    def __len__(self):
        return self.tensor.size(0)
