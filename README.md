# Transformer Architecture: Implementation from Scratch

This repository contains a comprehensive implementation of the Transformer architecture from scratch. The project includes data preprocessing, model training, and deployment for natural language understanding tasks.

## 📌 Getting Started

### 1️⃣ Clone the Repository
To begin, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/Sottim/transformer_implementation_scratch.git
cd transformer_arch
```

---

### 2️⃣ Environment Setup
#### Creating and Activating the Conda Environment
Ensure all required dependencies are installed by creating a Conda environment:

```bash
conda env create -f environment.yml
conda activate llm
```

#### 🔄 Updating the Environment
If new dependencies are added, update the environment using:

```bash
conda env update --file environment.yml --prune
```

---

### 3️⃣ Verifying Installation
Confirm successful installation by running the provided Jupyter notebooks:

```bash
jupyter notebook notebooks/01.ipynb
jupyter notebook notebooks/02.ipynb
```

---

## 📂 Dataset Preparation

### 🔹 Creating the Data Directory
Before processing data, create a directory to store the training dataset:

```bash
mkdir processed_data
```

This directory will store the processed training dataset.

### 🔹 Running Preprocessing Scripts

#### 🏗 Step 1: Run `data.py`
This script downloads and processes the **SQuAD V2 dataset**, formatting it into a Question-Answer structure.

```bash
python prepare_data/data.py
```

#### 🏗 Step 2: Run `append-nq-data.py`
This script appends the **Natural Questions (NQ) dataset** to the processed data.

```bash
python prepare_data/append-nq-data.py
```

#### 📌 Final Processed Dataset
After both scripts are executed, the processed dataset will be available at:

```
./processed_data/training_data_01.txt
```

---

## 🚀 Model Training
Initiate the training process by running the training script:

```bash
python src/train.py
```

#### 💾 Model Checkpoints & Weights
- Trained model weights are stored in the **`saved_model`** directory.
- Model checkpoints are stored in the **`checkpoints`** directory.

---

## 🎯 Running the Application
Deploy the trained model using the following command:

```bash
python app.py
```

Once executed, the application will be accessible at local host with port: 8000:

```
http://0.0.0.0:8000
```

---

## 📌 Notes
- Ensure all dependencies are installed before running the scripts.
- Modify `data.py` and `append-nq-data.py` to include additional datasets if needed.
- The processed dataset will serve as input for training the Transformer model.

---

