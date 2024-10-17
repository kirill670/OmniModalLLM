## English Version

# OmniModalLLM

OmniModalLLM is a versatile and powerful multimodal language model designed to handle both text and image inputs, enabling sophisticated conversational AI applications similar to ChatGPT. Leveraging advanced architectures like Mixture of Experts (MoE) and Vector Quantized Variational Autoencoders (VQVAE), OmniModalLLM offers robust performance and adaptability across various tasks.

## Features

- **Multimodal Support:** Handles both text and image data seamlessly.
- **Mixture of Experts (MoE):** Enhances model performance by leveraging multiple specialized experts.
- **Vector Quantized VAE (VQVAE):** Efficiently tokenizes and reconstructs images.
- **Dynamic Configuration:** Adjusts model components dynamically based on input data.
- **Chat API:** Provides a ChatGPT-like conversational interface using FastAPI.
- **Memory Optimizations:** Implements techniques like gradient checkpointing and mixed precision training to prevent CUDA Out-Of-Memory (OOM) errors.
- **Rate Limiting:** Protects the API from abuse using `slowapi`.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [API Deployment](#api-deployment)
- [API Endpoints](#api-endpoints)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA-enabled GPU (optional, for training and inference acceleration)

### Clone the Repository

```bash
git clone https://github.com/kirill670/OmniModalLLM.git
cd OmniModalLLM
```

### Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers datasets pillow fastapi uvicorn tiktoken einops tensorboard faiss-cpu slowapi
```

*Note:* If you're using a TPU or CPU, adjust the PyTorch installation accordingly.

## Usage

### Training

OmniModalLLM is pre-configured to train on the MS COCO dataset. Ensure you have sufficient computational resources before initiating training.

```bash
python main.py
```

This command will:

1. Load and preprocess the MS COCO dataset.
2. Initialize the OmniModalLLM model and tokenizer.
3. Start the training loop with mixed precision and gradient checkpointing.
4. Save model checkpoints upon improvement.
5. Launch the FastAPI server concurrently.

*Training Parameters:*

- **Epochs:** 5
- **Batch Size:** Adjusted based on available device (GPU, TPU, CPU)
- **Learning Rate:** 1e-4
- **Early Stopping:** Triggered after 3 epochs without improvement

### API Deployment

The FastAPI server provides a `/chat/` endpoint for interactive conversations. Once the training completes, the API server will be accessible at `http://0.0.0.0:8000/chat/`.

#### Running the API Server Independently

If you wish to run the API server without training, ensure the model is trained and load the saved checkpoint.

```bash
python main.py --load_checkpoint path_to_checkpoint.pth.tar
```

## API Endpoints

### POST `/chat/`

Generates a response based on the provided chat messages.

**Request Body:**

```json
{
  "session_id": "optional-session-id",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ]
}
```

- **`session_id`** (optional): Unique identifier to maintain conversation context across multiple requests. If not provided, a new session will be created.
- **`messages`**: List of message objects containing the role (`user`, `assistant`, or `system`) and the content.

**Response:**

```json
{
  "session_id": "unique-session-id",
  "message": {
    "role": "assistant",
    "content": "I'm a model designed to assist you. How can I help today?"
  }
}
```

## Examples

### Using `curl`

**Initial Request (Start a New Session):**

```bash
curl -X POST "http://localhost:8000/chat/" \
-H "Content-Type: application/json" \
-d '{
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ]
}'
```

**Subsequent Request (Continue the Conversation):**

```bash
curl -X POST "http://localhost:8000/chat/" \
-H "Content-Type: application/json" \
-d '{
    "session_id": "unique-session-id",
    "messages": [
        {"role": "user", "content": "Can you tell me a joke?"}
    ]
}'
```

### Using Postman

1. **Create a New POST Request:**
   - URL: `http://localhost:8000/chat/`
   
2. **Set Headers:**
   - `Content-Type`: `application/json`
   
3. **Set Body:**
   - Choose `raw` and `JSON` format.
   - Input the JSON payload as shown in the `curl` examples.
   
4. **Send the Request:**
   - Observe the assistant's reply in the response section.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the [MIT License](LICENSE).

---

## Russian Version

# OmniModalLLM

OmniModalLLM — это универсальная и мощная мультимодальная языковая модель, разработанная для обработки как текстовых, так и изображенческих данных. Она позволяет создавать сложные приложения для разговорного ИИ, аналогичные ChatGPT. Используя передовые архитектуры, такие как Mixture of Experts (MoE) и Vector Quantized Variational Autoencoders (VQVAE), OmniModalLLM обеспечивает высокую производительность и адаптивность для различных задач.

## Особенности

- **Мультимодальная поддержка:** Бесшовно обрабатывает как текстовые, так и изображенческие данные.
- **Mixture of Experts (MoE):** Улучшает производительность модели за счет использования нескольких специализированных экспертов.
- **Vector Quantized VAE (VQVAE):** Эффективно токенизирует и восстанавливает изображения.
- **Динамическая конфигурация:** Настраивает компоненты модели динамически на основе входных данных.
- **Чат API:** Предоставляет интерфейс для разговоров, похожий на ChatGPT, используя FastAPI.
- **Оптимизация памяти:** Реализует такие методы, как градиентный чекпоинтинг и обучение с смешанной точностью, чтобы избежать ошибок Out-Of-Memory (OOM) на CUDA.
- **Ограничение скорости запросов:** Защищает API от злоупотреблений с помощью `slowapi`.

## Содержание

- [Установка](#установка)
- [Использование](#использование)
  - [Обучение](#обучение)
  - [Развертывание API](#развертывание-api)
- [API Эндпоинты](#api-эндпоинты)
- [Примеры](#примеры)
- [Вклад](#вклад)
- [Лицензия](#лицензия)

## Установка

### Предварительные требования

- Python 3.8 или выше
- Git
- CUDA-совместимый GPU (опционально, для ускорения обучения и инференса)

### Клонирование репозитория

```bash
git clone https://github.com/kirill670/OmniModalLLM.git
cd OmniModalLLM
```

### Создание виртуального окружения

Рекомендуется использовать виртуальное окружение для управления зависимостями.

```bash
python -m venv venv
source venv/bin/activate  # В Windows: venv\Scripts\activate
```

### Установка зависимостей

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers datasets pillow fastapi uvicorn tiktoken einops tensorboard faiss-cpu slowapi
```

*Примечание:* Если вы используете TPU или CPU, настройте установку PyTorch соответствующим образом.

## Использование

### Обучение

OmniModalLLM преднастроена для обучения на датасете MS COCO. Убедитесь, что у вас есть достаточные вычислительные ресурсы перед началом обучения.

```bash
python main.py
```

Эта команда выполнит следующие действия:

1. Загрузит и предварительно обработает датасет MS COCO.
2. Инициализирует модель OmniModalLLM и токенизатор.
3. Запустит цикл обучения с использованием смешанной точности и градиентного чекпоинтинга.
4. Сохранит чекпоинты модели при улучшении результатов.
5. Одновременно запустит сервер FastAPI.

*Параметры обучения:*

- **Эпохи:** 5
- **Размер батча:** Настраивается в зависимости от доступного устройства (GPU, TPU, CPU)
- **Скорость обучения:** 1e-4
- **Ранняя остановка:** Активируется после 3 эпох без улучшения

### Развертывание API

Сервер FastAPI предоставляет эндпоинт `/chat/` для интерактивных разговоров. После завершения обучения API-сервер будет доступен по адресу `http://0.0.0.0:8000/chat/`.

#### Запуск API-сервера отдельно

Если вы хотите запустить API-сервер без обучения, убедитесь, что модель обучена и загрузите сохраненный чекпоинт.

```bash
python main.py --load_checkpoint path_to_checkpoint.pth.tar
```

## API Эндпоинты

### POST `/chat/`

Генерирует ответ на основе предоставленных сообщений чата.

**Тело запроса:**

```json
{
  "session_id": "optional-session-id",
  "messages": [
    {
      "role": "user",
      "content": "Привет, как дела?"
    }
  ]
}
```

- **`session_id`** (опционально): Уникальный идентификатор для поддержания контекста разговора между несколькими запросами. Если не предоставлен, будет создана новая сессия.
- **`messages`**: Список объектов сообщений, содержащих роль (`user`, `assistant` или `system`) и содержимое.

**Ответ:**

```json
{
  "session_id": "unique-session-id",
  "message": {
    "role": "assistant",
    "content": "Я модель, созданная для помощи вам. Чем могу помочь сегодня?"
  }
}
```

## Примеры

### Использование `curl`

**Начальный запрос (Создание новой сессии):**

```bash
curl -X POST "http://localhost:8000/chat/" \
-H "Content-Type: application/json" \
-d '{
    "messages": [
        {"role": "user", "content": "Привет, как дела?"}
    ]
}'
```

**Ответ:**

```json
{
  "session_id": "unique-session-id",
  "message": {
    "role": "assistant",
    "content": "Я модель, созданная для помощи вам. Чем могу помочь сегодня?"
  }
}
```

**Последующий запрос (Продолжение разговора):**

```bash
curl -X POST "http://localhost:8000/chat/" \
-H "Content-Type: application/json" \
-d '{
    "session_id": "unique-session-id",
    "messages": [
        {"role": "user", "content": "Расскажи анекдот."}
    ]
}'
```

**Ответ:**

```json
{
  "session_id": "unique-session-id",
  "message": {
    "role": "assistant",
    "content": "Почему пчёлы никогда не опаздывают? Потому что они всегда на улье!"
  }
}
```

### Использование Postman

1. **Создайте новый POST-запрос:**
   - URL: `http://localhost:8000/chat/`
   
2. **Установите заголовки:**
   - `Content-Type`: `application/json`
   
3. **Установите тело запроса:**
   - Выберите `raw` и формат `JSON`.
   - Введите JSON-пейлоад, как показано в примерах `curl`.
   
4. **Отправьте запрос:**
   - Наблюдайте ответ ассистента в разделе ответа.

### Создание простого фронтенда

Для более интерактивного опыта рассмотрите возможность создания простого фронтенда с использованием таких фреймворков, как React, Vue или даже простого HTML/CSS/JavaScript. Этот фронтенд может взаимодействовать с бэкендом FastAPI через эндпоинт `/chat/`.

## Вклад

Вклады приветствуются! Пожалуйста, следуйте этим шагам:

1. Форкните репозиторий.
2. Создайте новую ветку (`git checkout -b feature/YourFeature`).
3. Закоммитьте свои изменения (`git commit -m 'Добавить некоторую функцию'`).
4. Запушьте в ветку (`git push origin feature/YourFeature`).
5. Откройте Pull Request.

Пожалуйста, убедитесь, что ваш код соответствует стандартам кодирования проекта и включает соответствующие тесты.

## Лицензия

Этот проект лицензирован под [MIT License](LICENSE).

## Additional Information

For further assistance, questions, or suggestions, please feel free to open an issue on the [GitHub repository](https://github.com/kirill670/OmniModalLLM/issues).
