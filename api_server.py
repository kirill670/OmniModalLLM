# api_server.py

import os
import torch
from fastapi import FastAPI, UploadFile, File, Form, Request
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from PIL import Image
from transformers import LongformerTokenizer
from torch.utils.data import DataLoader, Dataset
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uuid

# Import the model and tokenizer from the training script or ensure they are accessible
from training_script import OmniModalLLM, LiquidFoundationTokenizer, device, conversation_history, generate_response

app = FastAPI()

# Initialize Limiter for Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize model and tokenizer
def initialize_model_and_tokenizer(device: torch.device):
    """
    Initializes the OmniModalLLM model and the LiquidFoundationTokenizer.

    Args:
        device (torch.device): The device to load the model onto.

    Returns:
        model (OmniModalLLM): The multimodal model.
        tokenizer (LiquidFoundationTokenizer): The tokenizer for processing text and images.
    """
    token_dim = 512
    channel_dim = 512
    expert_dim = 256    # Adjust as per training
    adapt_dim = 128     # Adjust as per training
    num_experts = 4     # Adjust as per training
    num_layers = 3      # Adjust as per training
    hidden_dim = 64
    num_heads = 8
    
    # Initialize the tokenizer
    tokenizer = LiquidFoundationTokenizer(device=device, adapt_dim=adapt_dim)
    
    # Initialize the model
    model = OmniModalLLM(
        token_dim=token_dim,
        channel_dim=channel_dim,
        expert_dim=expert_dim,
        adapt_dim=adapt_dim,
        num_experts=num_experts,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout_rate=0.1,
        max_drop_prob=0.1,
        layerdrop_prob=0.1,
        dropblock_block_size=7,
        dropblock_prob=0.1,
        combination_activation='gelu',
        combination_norm_type='batchnorm',
        norm_type='batchnorm',
        dynamic_layer_threshold=0.5
    ).to(device)
    
    # Load trained weights
    model.load_model('checkpoint.pth.tar')
    
    return model, tokenizer

model, tokenizer = initialize_model_and_tokenizer(device=device)

# API Models
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None  # To manage conversation sessions
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    session_id: str
    message: ChatMessage

# Conversation History Storage
# Ensure conversation_history is a thread-safe structure if accessed concurrently
from collections import defaultdict
import threading

conversation_history = defaultdict(list)
history_lock = threading.Lock()

# Inference Function
def generate_response_api(
    model: OmniModalLLM,
    tokenizer: LiquidFoundationTokenizer,
    user_text: str,
    session_id: str
) -> str:
    """
    Generates a response from the assistant based on user input and conversation history.

    Args:
        model (OmniModalLLM): The trained multimodal model.
        tokenizer (LiquidFoundationTokenizer): Tokenizer for processing text and images.
        user_text (str): The latest message from the user.
        session_id (str): Identifier for the conversation session.

    Returns:
        response_text (str): The assistant's generated response.
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Acquire lock to safely access conversation history
        with history_lock:
            history = conversation_history.get(session_id, []).copy()
        
        # Concatenate all user and assistant messages
        conversation = ""
        for msg in history:
            if msg.role == 'user':
                conversation += f"User: {msg.content}\n"
            elif msg.role == 'assistant':
                conversation += f"Assistant: {msg.content}\n"
        
        # Append the latest user message
        conversation += f"User: {user_text}\nAssistant:"
        
        # Tokenize the conversation
        tokenized = tokenizer.text_tokenizer.tokenize(conversation)
        tokens = tokenized['tokens'].unsqueeze(0).to(device)  # [1, seq]
        
        # Forward pass through the model
        outputs = model(tokens, image_embeddings=None)  # Assuming text-only for chat
        token_logits = outputs["token_logits"]  # [1, vocab_size]
        
        # Generate the assistant's response token
        predicted_token_id = torch.argmax(token_logits, dim=-1)  # [1]
        
        # Detokenize to get the response text
        response_text = tokenizer.text_tokenizer.detokenize(predicted_token_id)
        
        # Update conversation history
        with history_lock:
            conversation_history[session_id].append(
                ChatMessage(role="assistant", content=response_text)
            )
        
    return response_text.strip()

# API Endpoint
@app.post("/chat/", response_model=ChatResponse)
@limiter.limit("20/minute")  # Limit to 20 requests per minute per IP
async def chat_endpoint(request: ChatRequest, req: Request):
    """
    API endpoint to handle chat messages and generate responses.

    Args:
        request (ChatRequest): Incoming chat request containing messages and optional session_id.
        req (Request): The incoming HTTP request (used for rate limiting).

    Returns:
        ChatResponse: The assistant's response along with the session_id.
    """
    # Generate a new session_id if not provided
    if not request.session_id:
        session_id = str(uuid.uuid4())
        with history_lock:
            conversation_history[session_id] = []
    else:
        session_id = request.session_id
        with history_lock:
            if session_id not in conversation_history:
                conversation_history[session_id] = []
    
    # Append incoming messages to the conversation history
    with history_lock:
        conversation_history[session_id].extend(request.messages)
    
    # Extract the latest user message
    user_message = next((msg for msg in request.messages if msg.role == 'user'), None)
    if not user_message:
        return ChatResponse(
            session_id=session_id,
            message=ChatMessage(role="assistant", content="I didn't receive any user message.")
        )
    
    # Generate assistant's response
    assistant_reply = generate_response_api(
        model, 
        tokenizer, 
        user_message.content, 
        session_id=session_id
    )
    
    return ChatResponse(
        session_id=session_id,
        message=ChatMessage(role="assistant", content=assistant_reply)
    )

if __name__ == "__main__":
    # Launch the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)
