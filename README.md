# LLM-RAG: Retrieval-Augmented Generation Pipeline

A demo project for running a Retrieval-Augmented Generation (RAG) pipeline using Ollama, Qdrant, and Docker. This project loads technical documents, indexes them, and answers questions using a local LLM.

## Prerequisites

- [Git](https://docs.github.com/en/get-started/git-basics/set-up-git)
- [Docker](https://www.docker.com/get-started/)
- [Ollama](https://github.com/ollama/ollama/blob/main/README.md)
- [Ollama Model Library](https://ollama.com/library)

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/thismohsin/llm-rag.git
   cd llm-rag
   ```

[//]: # ()
[//]: # (2. **Build the Docker image:**)

[//]: # (   ```sh)

[//]: # (   docker build -t llm-rag-arm64-test .)

[//]: # (   docker images)

[//]: # (   ```)

3. **Start Ollama service:**
   ```sh
   docker compose up -d ollama
   docker ps
   ```

[//]: # (4. **Pull the required LLM model inside the Ollama container:**)

[//]: # (   ```sh)

[//]: # (   docker exec <container_id> ollama pull llama3.2:1b)

[//]: # (   ```)

[//]: # (   Replace `<container_id>` with the actual container ID from `docker ps`.)

5. **Start the RAG application:**
   ```sh
   docker compose up rag_app
   ```

6. **Verify Ollama model availability:**
   ```sh
   curl http://localhost:11434/api/tags
   ```

7. **Stop all services:**
   ```sh
   docker compose down
   ```

## Usage & Demo

- The pipeline will load documents from `docs/tech/`, index them, and answer sample questions.
- All output is printed to the console in a readable, demo-friendly format.
- No UI is required; simply follow the steps above and view the logs in your terminal.

## References
- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/README.md)
- [Ollama Model Library](https://ollama.com/library)
- [Qdrant Documentation](https://qdrant.tech/documentation/)

---

For troubleshooting, check Docker logs and ensure all services are running. For more details, see the documentation links above.
