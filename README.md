# OmniModalLLM: An Omnimodal Large Language Model for Text and Image Data

OmniModalLLM is a PyTorch-based model designed to process and generate responses from multimodal inputs, specifically combining text and image data. It incorporates advanced techniques like adaptive configuration, mixture of experts, and dynamic layers to produce an adaptable and efficient architecture. This model is suitable for tasks that require handling both text and image data simultaneously, and it includes a FastAPI server for real-time inference.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Architecture](#architecture)
- [Training](#training)
- [Inference and API Usage](#inference-and-api-usage)
- [Configuration](#configuration)
- [License](#license)

## Features

- **Multimodal Input Support**: Accepts both text and image data, and combines them for response generation.
- **Adaptive Layer Configuration**: Uses dynamic layers with adjustable weights based on input, making the model highly adaptable.
- **Mixture of Experts**: Selects from multiple sub-models based on attention mechanisms, improving the ability to generalize.
- **Dropout Regularization Techniques**: Includes advanced regularization methods such as DropPath, DropBlock, and LayerDrop.
- **API Server**: A FastAPI server for real-time response generation from text and image inputs.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/OmniModalLLM.git
   cd OmniModalLLM
   ```

2. **Install the dependencies**:

   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
   pip install transformers datasets pillow gradio fastapi uvicorn tiktoken einops tensorboard faiss-cpu
   ```

3. **Download Pre-trained Model Weights (Optional)**:
   If you have pre-trained model weights, place them in the root directory or specify their path in the code.

## Architecture

### Key Components

1. **LiquidVAE**: A variational autoencoder with adaptive linear layers for latent representation of text.
2. **Mixture of Experts**: Consists of several `LiquidLinear` experts combined with attention-based gating.
3. **Component Combination**: Dynamically combines token, channel, and expert outputs into a unified representation.
4. **LFModel (Layer Fusion Model)**: Core model with multiple layers that use token mixers, channel mixers, and attention.
5. **Adaptive Configuration**: Dynamically configures model layers based on input data.
6. **OmniModalLLM**: Main class integrating all components, handling multimodal input and generating responses.

## Training

1. **Load the Dataset**: The code uses the MS COCO dataset with captions and images as an example. Ensure you have access to this dataset or configure your dataset.

2. **Run Training**:
   ```bash
   python main.py
   ```

   During training:
   - The model is trained using a custom loss function that includes both token prediction and VAE reconstruction losses.
   - Mixed precision and gradient checkpointing are enabled for efficient training.

## Inference and API Usage

The model can be deployed via FastAPI to provide an inference endpoint.

1. **Start the API Server**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Access the API**:
   Send a `POST` request to `http://localhost:8000/generate/` with text and image data.

   Example request in Python:

   ```python
   import requests

   url = "http://localhost:8000/generate/"
   files = {'image': open('sample_image.jpg', 'rb')}
   data = {'text': "Your input text here"}

   response = requests.post(url, files=files, data=data)
   print(response.json())
   ```

3. **Response**:
   The API will return a generated response based on the provided text and image.

## Configuration

### Hyperparameters
The following hyperparameters can be modified within the code:
- `token_dim`, `channel_dim`, `expert_dim`, `adapt_dim`: Dimensions for various embeddings and adaptive layers.
- `num_experts`, `num_layers`: Number of experts and layers in the model.
- `dropout_rate`, `max_drop_prob`: Dropout and regularization settings.
- `combination_activation`, `norm_type`: Activation and normalization settings for layer combinations.

### Checkpoints
The model saves checkpoints during training. Update `save_path` in the code to specify where checkpoints should be saved.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
