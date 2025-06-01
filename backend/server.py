import os
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import asyncio
from datetime import datetime
import uuid
import yaml
import ast
import re
from pathlib import Path
import tempfile
import shutil
import base64

# Third-party imports for RAG
import tiktoken

# Lazy imports - will be loaded during startup
qdrant_client = None
SentenceTransformer = None
Distance = None
VectorParams = None
PointStruct = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Code Assistant", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
QDRANT_URL = "localhost"
QDRANT_PORT = 6333
DEFAULT_MODEL = "codellama:7b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Global clients
qdrant_client_instance = None
embedding_model = None

# Pydantic models
class GitLabConfig(BaseModel):
    gitlab_url: str = "https://gitlab.com"
    api_token: str
    repository_path: str  # e.g., "username/repo-name"
    branch: str = "main"

class RepositoryProcessRequest(BaseModel):
    config: GitLabConfig

class CodeSuggestionRequest(BaseModel):
    query: str
    context: Optional[str] = None
    suggestion_type: str = "general"  # general, bugfix, feature

class CodeChunk(BaseModel):
    id: str
    content: str
    file_path: str
    chunk_type: str  # ansible_task, python_function, yaml_config, etc.
    metadata: Dict[str, Any]

class ProcessingStatus(BaseModel):
    status: str
    message: str
    processed_files: int = 0
    total_files: int = 0

# Global status tracking
processing_status = ProcessingStatus(status="idle", message="Ready to process repository")

@app.on_event("startup")
async def startup_event():
    """Initialize clients on startup"""
    global qdrant_client_instance, embedding_model
    global qdrant_client, SentenceTransformer, Distance, VectorParams, PointStruct
    
    try:
        # Lazy import of dependencies
        import qdrant_client as qc
        from qdrant_client.models import Distance as Dist, VectorParams as VP, PointStruct as PS
        from sentence_transformers import SentenceTransformer as ST
        
        # Assign to global variables
        qdrant_client = qc
        Distance = Dist
        VectorParams = VP
        PointStruct = PS
        SentenceTransformer = ST
        
        # Initialize Qdrant client (in-memory mode)
        qdrant_client_instance = qdrant_client.QdrantClient(":memory:")
        logger.info("Connected to Qdrant (in-memory)")
        
        # Initialize embedding model
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
        
        # Create collection if it doesn't exist
        try:
            collections = qdrant_client_instance.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if "code_chunks" not in collection_names:
                qdrant_client_instance.create_collection(
                    collection_name="code_chunks",
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
                logger.info("Created code_chunks collection")
        except Exception as e:
            logger.warning(f"Collection setup warning: {e}")
            
    except Exception as e:
        logger.error(f"Startup error: {e}")

async def test_ollama_connection():
    """Test OLLAMA connection"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                return {"status": "connected", "models": available_models}
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def test_gitlab_connection(config: GitLabConfig):
    """Test GitLab API connection"""
    try:
        headers = {"Authorization": f"Bearer {config.api_token}"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test API access
            response = await client.get(f"{config.gitlab_url}/api/v4/user", headers=headers)
            if response.status_code == 200:
                user_info = response.json()
                
                # Test repository access
                repo_response = await client.get(
                    f"{config.gitlab_url}/api/v4/projects/{config.repository_path.replace('/', '%2F')}",
                    headers=headers
                )
                if repo_response.status_code == 200:
                    repo_info = repo_response.json()
                    return {
                        "status": "connected",
                        "user": user_info.get("username"),
                        "repository": repo_info.get("name"),
                        "access_level": "success"
                    }
                else:
                    return {"status": "error", "message": "Repository not accessible"}
            else:
                return {"status": "error", "message": "Invalid API token"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class AnsibleCodeParser:
    """Parse Ansible roles and Python modules"""
    
    @staticmethod
    def parse_ansible_task(content: str, file_path: str) -> List[CodeChunk]:
        """Parse Ansible YAML tasks"""
        chunks = []
        try:
            data = yaml.safe_load(content)
            if isinstance(data, list):
                for i, task in enumerate(data):
                    if isinstance(task, dict) and 'name' in task:
                        chunk_id = str(uuid.uuid4())
                        metadata = {
                            "task_name": task.get("name", ""),
                            "module": next((k for k in task.keys() if k not in ['name', 'tags', 'when', 'vars']), "unknown"),
                            "tags": task.get("tags", []),
                            "file_path": file_path,
                            "task_index": i
                        }
                        chunks.append(CodeChunk(
                            id=chunk_id,
                            content=yaml.dump(task, default_flow_style=False),
                            file_path=file_path,
                            chunk_type="ansible_task",
                            metadata=metadata
                        ))
        except yaml.YAMLError as e:
            logger.warning(f"YAML parsing error in {file_path}: {e}")
        return chunks
    
    @staticmethod
    def parse_python_module(content: str, file_path: str) -> List[CodeChunk]:
        """Parse Python modules"""
        chunks = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function with context
                    lines = content.split('\n')
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
                    
                    function_content = '\n'.join(lines[start_line:end_line])
                    
                    chunk_id = str(uuid.uuid4())
                    metadata = {
                        "function_name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "decorators": [ast.unparse(d) for d in node.decorator_list] if node.decorator_list else [],
                        "file_path": file_path,
                        "line_start": start_line + 1,
                        "line_end": end_line
                    }
                    
                    chunks.append(CodeChunk(
                        id=chunk_id,
                        content=function_content,
                        file_path=file_path,
                        chunk_type="python_function",
                        metadata=metadata
                    ))
                        
                elif isinstance(node, ast.ClassDef):
                    # Extract class definition
                    lines = content.split('\n')
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20
                    
                    class_content = '\n'.join(lines[start_line:end_line])
                    
                    chunk_id = str(uuid.uuid4())
                    metadata = {
                        "class_name": node.name,
                        "bases": [ast.unparse(base) for base in node.bases],
                        "file_path": file_path,
                        "line_start": start_line + 1,
                        "line_end": end_line
                    }
                    
                    chunks.append(CodeChunk(
                        id=chunk_id,
                        content=class_content,
                        file_path=file_path,
                        chunk_type="python_class",
                        metadata=metadata
                    ))
                        
        except SyntaxError as e:
            logger.warning(f"Python parsing error in {file_path}: {e}")
        return chunks

async def fetch_repository_contents(config: GitLabConfig) -> List[Dict[str, Any]]:
    """Fetch repository contents from GitLab"""
    headers = {"Authorization": f"Bearer {config.api_token}"}
    all_files = []
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Get repository tree
        url = f"{config.gitlab_url}/api/v4/projects/{config.repository_path.replace('/', '%2F')}/repository/tree"
        params = {"ref": config.branch, "recursive": "true", "per_page": "100"}
        
        response = await client.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to fetch repository tree: {response.text}")
        
        tree = response.json()
        
        # Filter for Ansible and Python files
        relevant_files = []
        for item in tree:
            if item["type"] == "blob":
                file_path = item["path"]
                if (file_path.endswith(('.yml', '.yaml')) or 
                    file_path.endswith('.py') or
                    'tasks/' in file_path or 
                    'handlers/' in file_path or
                    'vars/' in file_path):
                    relevant_files.append(item)
        
        # Fetch file contents
        for file_info in relevant_files:
            try:
                file_url = f"{config.gitlab_url}/api/v4/projects/{config.repository_path.replace('/', '%2F')}/repository/files/{file_info['path'].replace('/', '%2F')}"
                file_params = {"ref": config.branch}
                
                file_response = await client.get(file_url, headers=headers, params=file_params)
                if file_response.status_code == 200:
                    file_data = file_response.json()
                    content = base64.b64decode(file_data["content"]).decode("utf-8")
                    
                    all_files.append({
                        "path": file_info["path"],
                        "content": content,
                        "size": file_data.get("size", 0)
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to fetch {file_info['path']}: {e}")
                continue
    
    return all_files

async def process_repository_background(config: GitLabConfig):
    """Background task to process repository"""
    global processing_status
    
    try:
        processing_status.status = "fetching"
        processing_status.message = "Fetching repository contents..."
        
        # Fetch repository contents
        files = await fetch_repository_contents(config)
        processing_status.total_files = len(files)
        
        processing_status.status = "parsing"
        processing_status.message = "Parsing and chunking code..."
        
        # Parse and chunk files
        all_chunks = []
        parser = AnsibleCodeParser()
        
        for i, file_data in enumerate(files):
            try:
                file_path = file_data["path"]
                content = file_data["content"]
                
                if file_path.endswith(('.yml', '.yaml')):
                    if 'tasks/' in file_path or 'handlers/' in file_path:
                        chunks = parser.parse_ansible_task(content, file_path)
                    else:
                        # Handle vars, meta files as single chunks
                        chunk_id = str(uuid.uuid4())
                        chunks = [CodeChunk(
                            id=chunk_id,
                            content=content,
                            file_path=file_path,
                            chunk_type="yaml_config",
                            metadata={"file_type": "configuration", "file_path": file_path}
                        )]
                elif file_path.endswith('.py'):
                    chunks = parser.parse_python_module(content, file_path)
                else:
                    chunks = []
                
                all_chunks.extend(chunks)
                processing_status.processed_files = i + 1
                
            except Exception as e:
                logger.error(f"Error processing {file_data['path']}: {e}")
                continue
        
        processing_status.status = "embedding"
        processing_status.message = "Generating embeddings..."
        
        # Generate embeddings and store in Qdrant
        if all_chunks:
            points = []
            for chunk in all_chunks:
                try:
                    # Create text for embedding
                    embed_text = f"File: {chunk.file_path}\nType: {chunk.chunk_type}\nContent:\n{chunk.content}"
                    
                    # Generate embedding
                    embedding = embedding_model.encode(embed_text).tolist()
                    
                    # Create point for Qdrant
                    point = PointStruct(
                        id=chunk.id,
                        vector=embedding,
                        payload={
                            "content": chunk.content,
                            "file_path": chunk.file_path,
                            "chunk_type": chunk.chunk_type,
                            "metadata": chunk.metadata
                        }
                    )
                    points.append(point)
                    
                except Exception as e:
                    logger.error(f"Error creating embedding for chunk {chunk.id}: {e}")
                    continue
            
            # Store in Qdrant
            if points:
                qdrant_client_instance.upsert(
                    collection_name="code_chunks",
                    points=points
                )
        
        processing_status.status = "completed"
        processing_status.message = f"Successfully processed {len(all_chunks)} code chunks from {len(files)} files"
        
    except Exception as e:
        processing_status.status = "error"
        processing_status.message = f"Error: {str(e)}"
        logger.error(f"Repository processing error: {e}")

async def get_code_suggestions(query: str, suggestion_type: str = "general", limit: int = 5) -> Dict[str, Any]:
    """Get code suggestions using RAG"""
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode(query).tolist()
        
        # Search in Qdrant
        search_results = qdrant_client_instance.search(
            collection_name="code_chunks",
            query_vector=query_embedding,
            limit=limit
        )
        
        # Build context
        context_chunks = []
        for result in search_results:
            chunk_info = {
                "content": result.payload["content"],
                "file_path": result.payload["file_path"],
                "chunk_type": result.payload["chunk_type"],
                "score": result.score,
                "metadata": result.payload["metadata"]
            }
            context_chunks.append(chunk_info)
        
        # Create prompt based on suggestion type
        if suggestion_type == "bugfix":
            prompt = f"""You are an expert Ansible and Python developer. Based on the following code context, help identify and fix the bug described in this query: {query}

Relevant Code Context:
{format_context_for_prompt(context_chunks)}

Please provide:
1. Analysis of the potential bug
2. Specific code fix with explanations
3. Best practices to prevent similar issues

Response:"""
        elif suggestion_type == "feature":
            prompt = f"""You are an expert Ansible and Python developer. Based on the following code context, help implement the new feature described in this query: {query}

Relevant Code Context:
{format_context_for_prompt(context_chunks)}

Please provide:
1. Implementation approach
2. Code examples following existing patterns
3. Integration considerations with existing code

Response:"""
        else:
            prompt = f"""You are an expert Ansible and Python developer. Based on the following code context, provide helpful suggestions for this query: {query}

Relevant Code Context:
{format_context_for_prompt(context_chunks)}

Please provide relevant code suggestions, explanations, and best practices.

Response:"""
        
        # Call OLLAMA
        async with httpx.AsyncClient(timeout=120.0) as client:
            ollama_response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": DEFAULT_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "top_k": 40
                    }
                }
            )
            
            if ollama_response.status_code == 200:
                ollama_result = ollama_response.json()
                suggestion = ollama_result.get("response", "No response generated")
                
                return {
                    "suggestion": suggestion,
                    "context_chunks": context_chunks,
                    "query": query,
                    "suggestion_type": suggestion_type
                }
            else:
                raise HTTPException(status_code=500, detail="OLLAMA request failed")
                
    except Exception as e:
        logger.error(f"Code suggestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def format_context_for_prompt(context_chunks: List[Dict]) -> str:
    """Format context chunks for LLM prompt"""
    formatted = []
    for i, chunk in enumerate(context_chunks, 1):
        formatted.append(f"""
--- Context {i} (Score: {chunk['score']:.3f}) ---
File: {chunk['file_path']}
Type: {chunk['chunk_type']}
Content:
{chunk['content']}
""")
    return "\n".join(formatted)

# API Routes
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/api/status")
async def get_status():
    """Get current processing status"""
    return processing_status

@app.post("/api/test-connections")
async def test_connections(config: GitLabConfig):
    """Test all external connections"""
    results = {}
    
    # Test OLLAMA
    results["ollama"] = await test_ollama_connection()
    
    # Test GitLab
    results["gitlab"] = await test_gitlab_connection(config)
    
    # Test Qdrant
    try:
        collections = qdrant_client_instance.get_collections()
        results["qdrant"] = {"status": "connected", "collections": len(collections.collections)}
    except Exception as e:
        results["qdrant"] = {"status": "error", "message": str(e)}
    
    return results

@app.post("/api/process-repository")
async def process_repository(request: RepositoryProcessRequest, background_tasks: BackgroundTasks):
    """Start processing repository in background"""
    global processing_status
    
    if processing_status.status == "processing":
        raise HTTPException(status_code=400, detail="Repository processing already in progress")
    
    processing_status = ProcessingStatus(status="starting", message="Initializing repository processing...")
    background_tasks.add_task(process_repository_background, request.config)
    
    return {"message": "Repository processing started", "status": processing_status}

@app.post("/api/suggest")
async def suggest_code(request: CodeSuggestionRequest):
    """Get code suggestions"""
    if processing_status.status != "completed":
        raise HTTPException(status_code=400, detail="Repository not processed yet")
    
    result = await get_code_suggestions(request.query, request.suggestion_type)
    return result

@app.get("/api/repository-info")
async def get_repository_info():
    """Get information about processed repository"""
    try:
        # Get collection info
        collection_info = qdrant_client_instance.get_collection("code_chunks")
        
        # Get some sample chunks
        search_results = qdrant_client_instance.scroll(
            collection_name="code_chunks",
            limit=10
        )
        
        file_types = {}
        chunk_types = {}
        
        for point in search_results[0]:
            chunk_type = point.payload.get("chunk_type", "unknown")
            file_path = point.payload.get("file_path", "")
            
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            if file_path:
                ext = Path(file_path).suffix or "no_extension"
                file_types[ext] = file_types.get(ext, 0) + 1
        
        return {
            "total_chunks": collection_info.points_count,
            "chunk_types": chunk_types,
            "file_types": file_types,
            "status": processing_status.status
        }
        
    except Exception as e:
        logger.error(f"Repository info error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
