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

## 2️⃣ Environment Setup

You can set up the environment using either Conda (traditional method) or Docker (portable container method). Choose one based on your preference.

### **Option A: Conda Environment Setup**
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

### **Option B: Docker Setup**

#### 🛠 Prerequisites
- Install Docker on your system.

#### Building the Docker Image
Build the Docker image containing the application and its dependencies:

```bash
docker build -t transformer-arch-app .
```

This creates a portable container with a CPU-only setup, using **Python 3.9, PyTorch 2.3.1, and TorchVision 0.18.1**.

#### 🚀 Running the Docker Container
Run the application in a container:

```bash
docker run -p 8000:8000 transformer-app
```

The application will be accessible at **http://localhost:8000**.

---

## 3️⃣ Verifying Installation

### **Conda Method**
Confirm successful installation by running the provided Jupyter notebooks:

```bash
jupyter notebook notebooks/01.ipynb
jupyter notebook notebooks/02.ipynb
```

### **Docker Method**
After running the container, open a browser and navigate to **http://localhost:8000** to verify the application is running.

---

## 📂 Dataset Preparation

### 🔹 Creating the Data Directory
Before processing data, create a directory to store the training dataset:

```bash
mkdir processed_data
```

This directory will store the processed training dataset.

### 🔹 Running Preprocessing Scripts

**Note:** These steps require the **Conda environment (`llm`)** to be active, as they involve Python scripts not included in the Docker container.

#### Step 1: Run `data.py`
This script downloads and processes the **SQuAD V2 dataset**, formatting it into a Question-Answer structure.

```bash
python prepare_data/data.py
```

#### Step 2: Run `append-nq-data.py`
This script appends the **Natural Questions (NQ) dataset** to the processed data.

```bash
python prepare_data/append-nq-data.py
```

#### Final Processed Dataset
After both scripts are executed, the processed dataset will be available at:

```
./processed_data/training_data_01.txt
```

---

## 🚀 Model Training

**Note:** Training requires the **Conda environment (`llm`)** to be active, as the Docker setup is for **deployment only**.

Initiate the training process by running the training script:

```bash
python src/train.py
```

### 💾 Model Checkpoints & Weights
- Trained model weights are stored in the **`save_model`** directory.
- Model checkpoints are stored in the **`checkpoints`** directory.

---

## 🎯 Running the Application

### **Using Conda**
Deploy the trained model using:

```bash
python app.py
```

### **Using Docker**
Build and run the Docker container (see **Docker Setup** above):

```bash
docker build -t transformer-app .
docker run -p 8000:8000 transformer-app
```

Once executed (via either method), the application will be accessible at:

```
http://localhost:8000
```

---

## 📌 Notes

- **Conda:** Ensure all dependencies are installed before running scripts. Modify `data.py` and `append-nq-data.py` to include additional datasets if needed.
- **Docker:** The container includes only the runtime files (`app.py`, `frontend/`, `save_model/`, `src/`) and a **CPU-only setup** for portability. Training and preprocessing are **not** supported in the Docker image.
- The processed dataset (`processed_data/`) serves as input for training the Transformer model, while `save_model/` contains the weights used by `app.py`.

