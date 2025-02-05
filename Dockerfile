FROM ubuntu:22.04

# Install dependencies (without sudo)
RUN apt update && apt install -y curl

# Install Ollama inside the container
RUN curl -fsSL https://ollama.com/install.sh | sh

# Ensure the Ollama model directory exists
RUN mkdir -p /root/.ollama

# Expose Ollama API port
EXPOSE 11111

# Set Ollama to listen on all network interfaces
ENV OLLAMA_HOST=0.0.0.0:11111

# Start Ollama, wait a bit, then pull the model, then keep Ollama running
CMD (ollama serve &) && sleep 5 && ollama pull llama3 && wait
