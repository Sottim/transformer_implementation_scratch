import torch 
from torch.utils.data import Dataset
from transformer import GPT2Tokenizer

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
        self.qa_pairs = [] # List to store question-answer pairs

        # Extract question-answer pairs
        for pair in pairs:
            try:
                # Splits at "Question: ", taking everything after it. Further splits at "\nAnswer: ", extracting only the question text.
                question = pair.split('Question: ')[1].split('\nAnswer: ')[0] 
                answer = pair.split('Answer: ')[1]
                self.qa_pairs.append((question, answer))
                # print(f"QA: {self.qa_pairs}") # Debugging

            except:
                continue #If a pair doesn’t follow the expected format, it skips that entry.

    def __len__(self):
        return len(self.qa_pairs) # Number of question-answer pairs
    
    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        
        # Format as: "Question: {question} Answer: {answer}"
        input_text = f"Question: {question} Answer: "
        target_text = answer
        
        # Tokenize input
        input_encodings = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encodings = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encodings['input_ids'].squeeze(),
            'attention_mask': input_encodings['attention_mask'].squeeze(),
            'target_ids': target_encodings['input_ids'].squeeze(),
            'target_attention_mask': target_encodings['attention_mask'].squeeze()
        }
    
"""Example output of the dataset:
{
    'input_ids': tensor([32, 521, 123, ...]),   # Question tokens
    'attention_mask': tensor([1, 1, 1, ...]),   # 1 for real tokens, 0 for padding
    'target_ids': tensor([456, 789, 12, ...]),  # Answer tokens
    'target_attention_mask': tensor([1, 1, 1, ...])
}
"""

# # Initialize tokenizer and dataset
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# """
# Problem: GPT-2 Has No Native Padding Token
#     GPT-2 is a causal (autoregressive) language model i.e GPT-2 was designed to generate text one word at a time (like autocomplete), so it doesn’t need padding in normal use., so it does not require padding during normal inference.
#     However, during fine-tuning, we need to use batch training, where all sequences in a batch must have the same length.
#     To achieve this, we pad shorter sequences to the same length. 
# """
# tokenizer.pad_token = tokenizer.eos_token 

# train_dataset = TextDataset('processed_data/training_data_02.txt', tokenizer)
# print(len(train_dataset)) # Number of question-answer pairs
# print(train_dataset[0]) # First question-answer pair





