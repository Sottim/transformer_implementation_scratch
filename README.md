# Transformer Architecture Project

## Cloning the Project

To get started, clone this repository using:

```bash
git clone https://github.com/Sottim/transformer_implementation_scratch.git
cd transformer_arch
```

## Setup Instructions

### 1. Setting Up the Environment

Set up the environment using Conda:

```bash
conda env create -f environment.yml
```

Then, activate the environment:

```bash
conda activate llm
```

### Updating the Environment
If you add more dependencies later, update `environment.yml` and run:

```bash
conda env update --file environment.yml --prune
```


### 2. Verify Installation
Ensure the installation is successful by running the provided Python notebooks in notebooks direcotry:

```bash
notebook 01.ipynb
notebook 02.ipynb
```

### 3. Create Data Directory
Before proceeding, create a directory to store training data:

```bash
mkdir processed_data
```

Inside the `processed_data` directory, you will write the `training_data.txt` file.

---

## Dataset Preprocessing & Processing

The `prepare_data` directory contains two scripts responsible for downloading and processing datasets.

### **1. Run `data.py`**

This script downloads and processes the **SQuAD V2 dataset** and stores it in `./processed_data/training_data_01.txt` in **Question-Answer format**.

```bash
python prepare_data/data.py
```

### **2. Run `append-nq-data.py`**

This script downloads the **Natural Questions (NQ) dataset** and appends it to `./processed_data/training_data_01.txt`.

```bash
python prepare_data/append-nq-data.py
```

### **Final Processed Dataset**
After running both scripts, the final dataset for training will be stored in:

```
./processed_data/training_data_01.txt
```

This file contains questions and answers formatted for model training.

---

## Model Training



## Notes
- Ensure that all dependencies are installed before running the scripts.
- Modify `data.py` and `append-nq-data.py` if you need to process additional datasets.
- The processed dataset will be used for training the transformer model.

Happy Coding! 🚀

