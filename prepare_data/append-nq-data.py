from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import json
import os
import random

# Append the Natural Questions dataset to the existing training_data_01.txt file
def append_nq_data(output_path="training_data_02.txt", max_examples=100000):
    print("Downloading and preparing datasets...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Loading Natural Questions dataset...")
    nq_dataset = load_dataset("natural_questions", split="train", streaming=True)
    # first_example = next(iter(nq_dataset))
    # print(first_example.keys())  # Print the keys of the first example  

    with open(output_path, 'a', encoding='utf-8') as f:
        qa_count = 0
        for item in tqdm(nq_dataset.take(max_examples), desc="Processing Natural Questions", total=max_examples):
            try:
                # Ensure the item contains necessary fields
                if "question" in item and "annotations" in item:
                    question = item["question"]['text']
                    # print(question) # Debugging output

                    tokens = item["question"].get('tokens', [])  # Extract tokens if available

                    # Extract annotations
                    annotations = item["annotations"]

                    # Directly extract the answer while ensuring it's not a list
                    try:
                        raw_answer = item["annotations"]['short_answers'][0]['text']  # Extract
                        answer = raw_answer[0] if isinstance(raw_answer, list) and raw_answer else raw_answer  # Get first element if it's a list
                    except (KeyError, IndexError, TypeError):
                        answer = None  # Handle missing fields safely

                    # Ensure valid Q&A pair
                    if question and answer and isinstance(answer, str) and len(question.split()) >= 3 and len(answer.split()) >= 1:
                        f.write(f"Question: {question}\nAnswer: {answer}\n\n")
                        qa_count += 1
                    else:
                        print(f"Skipping: Question = {question}, Answer = {answer}")  # Debugging output

            except Exception as e:
                print(f"Error processing item: {e}")  # Debugging output
                continue

    
    print(f"\nData preparation completed!")
    print(f"Processed {qa_count} Natural Questions examples")
    print(f"Appended dataset saved to: {output_path}")
    
    return output_path

# Get the dataset statistics
def get_dataset_stats(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        qa_pairs = content.count('Question: ')
        print("\nDetailed Dataset Statistics:")
        print(f"Number of Q&A pairs: {qa_pairs:,}")

        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)  # MB
        size_gb = size_mb / 1024  # GB

        size_bytes = os.path.getsize(file_path)
        print(f"File size: {size_bytes:,} bytes")
        print(f"         = {size_mb:.2f} MB")
        print(f"         = {size_gb:.3f} GB")


if __name__ == "__main__":
    # Append the Natural Questions dataset to the existing training_data_01.txt file
    data_path = append_nq_data(output_path="./processed_data/training_data_01.txt", max_examples=100000)
    # Get the dataset statistics
    get_dataset_stats(data_path)