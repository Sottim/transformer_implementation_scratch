from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import GPT2Tokenizer
from src.transformer import Transformer 
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from src.logger import log  # Import log function from logger.py

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

MODEL_PATH = "./save_model/best_model.pth"

num_layers = 12
num_heads = 12
d_model = 768
d_ff = 3072
dropout = 0.1
max_seq_len = 512

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    dropout=dropout
).to(device)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        log(f"Loading model state from: {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        log("Model and tokenizer loaded successfully!")
    except Exception as e:
        log(f"Error loading model: {str(e)}")
        raise
    yield

app.lifespan = lifespan

class Query(BaseModel):
    question: str

@app.post("/answer/")
async def answer_question(query: Query):
    question = f"Question: {query.question.strip()} Answer: "
    try:
        log(f"Processing question: {question}")
        input_ids = tokenizer.encode(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len
        ).to(device)
        log(f"Input IDs shape: {input_ids.shape}")
            
        with torch.no_grad():
            log("Calling model.generate...")
            output_ids = model.generate(
                src=input_ids,
                max_length=100,
                temperature=0.7,
                top_k=50,
                tokenizer=tokenizer
            )
            log(f"Output IDs: {output_ids}")
        
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        log(f"Generated text: {output_text}")
        
        if "Answer:" in output_text:
            answer = output_text.split("Answer:")[1].strip()
        else:
            answer = output_text.strip()
        
        return {"question": query.question, "answer": answer}
    except Exception as e:
        log(f"Error: {str(e)}")
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