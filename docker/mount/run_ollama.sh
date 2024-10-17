#!/bin/bash

echo "Starting Ollama server..."
ollama serve &  # Start Ollama in the background

echo "Ollama is ready, creating the model..."

ollama create qwen-anime2 -f mount/Modelfile
ollama run qwen-anime2