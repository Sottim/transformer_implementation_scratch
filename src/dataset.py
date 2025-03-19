import torch 
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

class TextDataset(Dataset):
    """
    Input:
    text_path → Path to the text file containing question-answer pairs.
    tokenizer → Tokenizer to encode text data.
    max_length → Maximum length of the input and target sequences.
    
    Returns:
    input_ids → Encoded question (Question: ... Answer: )
    attention_mask → Marks which tokens are padding (0) or real (1).
    target_ids → Encoded answer.
    target_attention_mask → Marks padding for the answer.
    
    """

    def __init__(self, text_path, tokenizer, max_length=512):
        self.text_path = text_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data from text file
        print(f"Loading data from {self.text_path} ...") 
        
        # Read the text file
        with open(self.text_path, 'r', encoding='utf-8') as f:
            content = f.read().strip() # remove leading/trailing whitespaces 

        pairs = content.split('\n\n') # split by double new line
        # print(f"Pairs : {pairs} ")
        self.sequences = []

        # Extract question-answer pairs
        for pair in pairs:
            try:
                # Splits at "Question: ", taking everything after it. Further splits at "\nAnswer: ", extracting only the question text.
                question = pair.split('Question: ')[1].split('\nAnswer: ')[0] 
                answer = pair.split('Answer: ')[1]
                full_sequence = f"Question: {question} Answer: {answer}"
                self.sequences.append(full_sequence)

            except:
                continue #If a pair doesn’t follow the expected format, it skips that entry.

    def __len__(self):
        return len(self.sequences) # Number of question-answer pairs
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # print(f"Sequence {idx}: {sequence}")  # Debug single sequence
        encodings = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].squeeze(0)
        # print(f"After squeeze, input_ids shape: {input_ids.shape}")  # Debug shape
        attention_mask = encodings['attention_mask'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # Labels are the same as input_ids, shifted during loss computation
        }
    
"""Example output of the dataset:
{
    'input_ids': tensor([32, 521, 123, ...]),   # Question tokens
    'attention_mask': tensor([1, 1, 1, ...]),   # 1 for real tokens, 0 for padding
}
"""

# # Testing this dataset class
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# """
# Problem: GPT-2 Has No Native Padding Token
#     GPT-2 is a causal (autoregressive) language model i.e GPT-2 was designed to generate text one word at a time (like autocomplete), so it doesn’t need padding in normal use., so it does not require padding during normal inference.
#     However, during fine-tuning, we need to use batch training, where all sequences in a batch must have the same length.
#     To achieve this, we pad shorter sequences to the same length. 
# """
# tokenizer.pad_token = tokenizer.eos_token 

# train_dataset = TextDataset('processed_data/training_data_02.txt', tokenizer)
# print(len(train_dataset)) # Number of sequences to be processed
# # print(train_dataset[0])

# # Check the first sequence
# print(f"Input IDs: {train_dataset[0]['input_ids']}")
# print(f"Attention Mask: {train_dataset[0]['attention_mask']}")
# print(f"Labels: {train_dataset[0]['labels']}")
