FROM python:3.9-bullseye

WORKDIR /app

# Set environment variables to disable CUDA
ENV CUDA_VISIBLE_DEVICES=""
ENV TORCH_CUDA_ARCH_LIST="None"
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32"
ENV USE_CUDA="0"
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=""
ENV TORCH_CUDA_ARCH_LIST="None"
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32"
# Use a larger volume for Hugging Face cache
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/hf_cache


# Install build essentials and clean up in same layer to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install compatible versions of core dependencies first
RUN pip install --no-cache-dir \
    "numpy>=1.21.0,<2.0.0" \
    "setuptools>=65.0.0"

# Install PyTorch and related packages (let pip select the correct wheel for ARM64)
RUN pip install --no-cache-dir \
    torch==2.1.0

# Install compatible transformers and sentence-transformers versions
RUN pip install --no-cache-dir \
    "transformers>=4.21.0,<4.36.0" \
    "tokenizers>=0.13.0,<0.15.0" \
    "sentence-transformers>=2.2.0,<2.8.0"

# Install main application dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    psycopg2-binary \
    "faiss-cpu>=1.7.0" \
    qdrant-client==1.6.3 \
    langchain==0.0.335 \
    langchain-community==0.0.13 \
    requests==2.31.0 \
    python-dotenv==1.0.0 \
    fastapi==0.103.1 \
    uvicorn==0.23.2 \
    pydantic==2.4.2 \
    scikit-learn==1.3.1

# Install evaluation dependencies (separate layer to avoid caching issues)
RUN pip install --no-cache-dir pandas==2.0.3 pyarrow==12.0.0 datasets==2.14.0 ragas==0.1.0

# Pre-download the model to avoid runtime download issues
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && rm -rf /app/hf_cache

# Copy project files
COPY services ./services
COPY docs ./docs
