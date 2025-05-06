import json
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class SupervisedDataset(Dataset):
    def __init__(self, jsonl_file: str, tokenizer: PreTrainedTokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # ✅ 补丁：确保 pad_token 设置好
        if self.tokenizer.pad_token is None:
            print("✅ tokenizer.pad_token 在 Dataset 中设置为 eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        prompt = example['prompt']
        response = example['response']
        full_text = prompt + "\n" + response

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
