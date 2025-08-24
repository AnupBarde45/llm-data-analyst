# Setup Ollama with Llama 3

## Install Ollama

1. Download from https://ollama.ai/
2. Install and start Ollama

## Pull Llama 3 Model

```bash
ollama pull llama3
```

## Start Ollama Server

```bash
ollama serve
```

## Test Connection

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt": "Hello",
  "stream": false
}'
```

Now you can run the data analyst assistant!