from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import GPT2Tokenizer
from src.transformer import Transformer 
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "*"],  # Allow all origins for simplicity; restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  
    allow_headers=["*"], 
)

# Mount the frontend directory for static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Paths to your trained model weights
MODEL_PATH = "./save_model/transformer_best.pth"

# Model hyperparameters (must match train.py)
batch_size = 16
num_layers = 6
num_heads = 8
d_model = 512
d_ff = 2048
dropout = 0.3
max_seq_len = 512

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
input_vocab_size = tokenizer.vocab_size
target_vocab_size = tokenizer.vocab_size

# Initialize model with the same architecture as in train.py
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    max_seq_len=max_seq_len,
    dropout=dropout
).to(device)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    yield

app.lifespan = lifespan

class Query(BaseModel):
    question: str

@app.post("/answer/")
async def answer_question(query: Query):
    question = f"Question: {query.question.strip()}"
    try:
        input_ids = tokenizer.encode(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len
        ).to(device)
        
        src_mask = (input_ids != tokenizer.pad_token_id).float().to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                src=input_ids,
                src_mask=src_mask,
                max_length=50,
                temperature=0.1,
                top_k=20,
                top_p=0.9,
                do_sample=False,
                tokenizer=tokenizer
            )
        
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer = output_text.replace(question, "").strip()
        if "Answer:" in output_text:
            answer = output_text.split("Answer:")[1].strip()
        
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.get("/", include_in_schema=False)
async def get_index():
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health_check():
    return {"status": "Question-Answering API is up and running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)