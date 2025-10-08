"""
Complete RAG pipeline test with Ollama for answer generation.
"""
import os
import traceback
import sys
import requests
import json
from document_processor import DocumentProcessor
from lightweight_embedder import LightweightEmbedder
from vector_store import QdrantVectorStore
from rag_evaluator import RAGEvaluator
import time
from qdrant_client.http.exceptions import ResponseHandlingException

class OllamaLLM:
    """Simple Ollama client for generating answers."""

    def __init__(self, base_url: str = "http://ollama:11434", model: str = "llama3.2:1b"):
        self.base_url = base_url
        self.model = model

    def check_model_available(self) -> bool:
        """Check if the model is available in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                print(f"Available models: {available_models}")
                return any(self.model in model_name for model_name in available_models)
            return False
        except Exception as e:
            print(f"Error checking models: {e}")
            return False

    def wait_for_model(self, max_wait_time: int = 300) -> bool:
        """Wait for the model to be available."""
        print(f"Waiting for {self.model} model to be available...")
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            if self.check_model_available():
                print(f"Model {self.model} is now available!")
                return True
            print(f"Model {self.model} not ready yet, waiting 10 seconds...")
            time.sleep(10)

        print(f"Timeout waiting for model {self.model}")
        return False

    def generate(self, prompt: str) -> str:
        """Generate text using Ollama."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60  # Increased timeout for model responses
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            elif response.status_code == 404:
                return f"Error: Model '{self.model}' not found. Available models can be checked with /api/tags"
            else:
                print(f"Ollama API error: {response.status_code}")
                print(f"Response: {response.text}")
                return f"Error: Could not generate response (status {response.status_code})"

        except requests.exceptions.RequestException as e:
            print(f"Failed to connect to Ollama: {e}")
            return f"Error: Could not connect to Ollama - {str(e)}"

def create_rag_prompt(question: str, contexts: list) -> str:
    """Create a RAG prompt with retrieved contexts."""
    context_text = "\n\n".join(contexts)

    prompt = f"""Based on the following context, please answer the question. If the answer is not in the context, say "I cannot answer based on the provided context."

Context:
{context_text}

Question: {question}

Answer:"""

    return prompt

class RAGPipeline:
    """RAG Pipeline that integrates with RAGEvaluator to avoid code duplication."""

    def __init__(self, qdrant_host: str = "qdrant", ollama_url: str = "http://ollama:11434",
                 ollama_model: str = "llama3.2:1b"):
        self.qdrant_host = qdrant_host
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

        # Initialize components
        self.processor = DocumentProcessor(docs_dir="/app/docs")
        self.embedder = LightweightEmbedder(max_features=384)
        self.vector_store = QdrantVectorStore(
            host=qdrant_host,
            collection_name="rag_test",
            vector_size=384
        )
        self.llm = OllamaLLM(base_url=ollama_url, model=ollama_model)

        # Initialize evaluator with the same components
        self.evaluator = RAGEvaluator(
            ground_truth_file="/app/docs/tech/tech_qa_pairs.json",
            embedder=self.embedder,
            vector_store=self.vector_store,
            qdrant_host=qdrant_host,
            collection_name="rag_test",
            ollama_base_url=ollama_url,
            ollama_model=ollama_model
        )

    def setup_and_index_documents(self):
        """Load and index documents."""
        print("\n1. Loading and indexing documents...")
        docs = self.processor.load_text_documents(subdirs=["tech"])
        print(f"Loaded {len(docs)} documents")

        # Wait for Qdrant to be ready
        print("Waiting for Qdrant to be ready...")
        time.sleep(3)

        # Index documents
        texts = [doc["content"] for doc in docs]
        embeddings = self.embedder.embed(texts)
        payloads = [{"text": doc["content"], "source": doc["metadata"].get("source")} for doc in docs]
        self.vector_store.upsert(embeddings, payloads)
        print(f"Indexed {len(docs)} documents in Qdrant")

        return docs

    def initialize_ollama(self):
        """Initialize and test Ollama connection."""
        print("\n2. Initializing Ollama...")

        # Wait for the model to be available
        if not self.llm.wait_for_model():
            print("Could not connect to Ollama or model not available. Exiting...")
            return False

        # Test Ollama connection
        test_response = self.llm.generate("Hello, respond with 'Ollama is working!'")
        print(f"Ollama test response: {test_response}")
        return True

    def test_rag_pipeline(self):
        """Test the RAG pipeline with sample questions."""
        print("\n3. Testing complete RAG pipeline...")

        test_questions = [
            "How do I list files in Linux?",
            "What command is used to change directories in Linux?",
            "How do I initialize a git repository?"
        ]

        generated_answers = []

        for i, question in enumerate(test_questions):
            print(f"\n--- Question {i+1}: {question} ---")

            # Step 1: Retrieve relevant contexts using evaluator's method
            print("Retrieving contexts...")
            contexts = self.evaluator.retrieve_context(question, top_k=2)

            print(f"Found {len(contexts)} relevant contexts:")
            for j, context in enumerate(contexts):
                print(f"Context {j+1}: {context[:100]}...")

            # Step 2: Generate answer using Ollama
            print("Generating answer with Ollama...")
            rag_prompt = create_rag_prompt(question, contexts)
            generated_answer = self.llm.generate(rag_prompt)

            print(f"Generated Answer: {generated_answer}")
            print("-" * 50)

            generated_answers.append(generated_answer)

        return test_questions, generated_answers

    def run_evaluation(self, questions: list, answers: list):
        """Run RAGAS evaluation using the integrated evaluator."""
        print("\n4. Running RAGAS evaluation...")

        try:
            results = self.evaluator.evaluate(
                questions=questions,
                answers=answers,
                model_name="ollama-llama3.2:1b"
            )
            self.evaluator.print_results(results)
            return results
        except Exception as e:
            print(f"RAGAS evaluation failed: {str(e)}")
            print("Continuing with simple ground truth comparison...")
            return None

    def test_ground_truth_comparison(self):
        """Test against ground truth for comparison."""
        print("\n5. Testing against ground truth...")

        # Load ground truth
        with open("/app/docs/tech/tech_qa_pairs.json", 'r') as f:
            qa_pairs = json.load(f)

        questions = []
        generated_answers = []

        for qa_pair in qa_pairs[:2]:  # Test first 2 pairs
            question = qa_pair["question"]
            ground_truth = qa_pair["answer"]

            print(f"\nQuestion: {question}")
            print(f"Ground Truth: {ground_truth}")

            # RAG pipeline using evaluator's retrieve method
            contexts = self.evaluator.retrieve_context(question, top_k=2)
            rag_prompt = create_rag_prompt(question, contexts)
            generated_answer = self.llm.generate(rag_prompt)

            print(f"RAG Answer: {generated_answer}")
            print("-" * 50)

            questions.append(question)
            generated_answers.append(generated_answer)

        return questions, generated_answers

def main():
    try:
        print("Initializing RAG pipeline with Ollama...")

        # Get environment variables
        qdrant_host = os.environ.get("QDRANT_HOST", "qdrant")
        ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")

        print(f"Qdrant host: {qdrant_host}")
        print(f"Ollama URL: {ollama_url}")

        # Initialize RAG pipeline with integrated evaluator
        rag_pipeline = RAGPipeline(
            qdrant_host=qdrant_host,
            ollama_url=ollama_url,
            ollama_model="llama3.2:1b"
        )

        # Setup and index documents
        docs = rag_pipeline.setup_and_index_documents()

        # Initialize Ollama
        if not rag_pipeline.initialize_ollama():
            return

        # Test RAG pipeline
        test_questions, test_answers = rag_pipeline.test_rag_pipeline()

        # Run RAGAS evaluation (if available)
        rag_pipeline.run_evaluation(test_questions, test_answers)

        # Test ground truth comparison
        gt_questions, gt_answers = rag_pipeline.test_ground_truth_comparison()

        # Run evaluation on ground truth questions too
        if gt_questions and gt_answers:
            print("\n6. Running RAGAS evaluation on ground truth questions...")
            rag_pipeline.run_evaluation(gt_questions, gt_answers)

        print("\nRAG pipeline test completed successfully!")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(traceback.format_exc())
        sys.stdout.flush()

if __name__ == "__main__":
    main()
