import os
import sys
# Add parent dir to path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.database import SessionLocal, Base, engine
from app.rag.ingestion import ingest_document

def create_sample_docs():
    sample_dir = "data/sample"
    os.makedirs(sample_dir, exist_ok=True)
    
    docs = {
        "auth_guide.txt": "The authentication mechanism uses OAuth 2.0 with JWT tokens.\n\nPass the token in the Authorization header as 'Bearer <token>'. Tokens expire in 60 minutes. To get a new token, use the refresh token endpoint.\n\nAll external integrations must authenticate via the gateway proxy using their provisioned client ID and client secret to obtain this token. User sessions are tracked via Redis to prevent token reuse upon logout.",
        "api_docs.txt": "Our REST API endpoints return JSON data. The base URL is https://api.example.com.\n\nFor rate limiting, the limit is 100 requests per minute per user. HTTP 429 means Too Many Requests. The API adheres to strict RESTful conventions.\n\nPagination is handled using 'limit' and 'offset' query parameters. Clients should check the 'X-Total-Count' header in the response. Webhooks are available for real-time state synchronization.",
        "deployment.txt": "To deploy the application, run 'docker-compose up -d'.\n\nEnsure that the environment variables in .env are correctly set. The database uses PostgreSQL on port 5432. Redis is required on port 6379 for caching.\n\nFor production, we recommend Kubernetes via the provided Helm charts. Node pools should be configured with auto-scaling to handle peak load. Secrets must be injected using a secure vault rather than environment variables.",
        "security_policy.txt": "All passwords must be at least 12 characters, contain upper, lower, numbers, and symbols.\n\nWe audit access logs every 30 days. No hardcoded secrets in the repository. Two-factor authentication (2FA) is mandatory for all administrative accounts.\n\nAny discovered vulnerabilities must be reported through the bug bounty portal. Data at rest is encrypted using AES-256 standard. TLS 1.3 is strictly enforced for all inbound HTTP traffic.",
        "troubleshooting.txt": "If the app fails to start, check the database connection.\n\nThe most common error is 'Connection refused' which means the PostgreSQL container is not running or credentials are wrong.\n\nIf you encounter high latency, inspect the FAISS vector index memory consumption. Ensure that enough RAM is allocated to the Docker container. For unexpected 500 errors, always check the uvicorn error logs in the /var/log directory."
    }
    
    for filename, content in docs.items():
        with open(os.path.join(sample_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)
            
    return docs.keys()

def seed_data():
    print("Creating tables...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    print("Creating sample documents...")
    filenames = create_sample_docs()
    
    print("Ingesting documents (chunking, embedding, indexing)...")
    db = SessionLocal()
    try:
        for filename in filenames:
            file_path = os.path.join("data/sample", filename)
            ingest_document(file_path, filename, "text/plain", db)
            print(f"Ingested {filename}")
    except Exception as e:
        print(f"Error seeding data: {e}")
    finally:
        db.close()
        
    print("Seed complete!")

if __name__ == "__main__":
    seed_data()
