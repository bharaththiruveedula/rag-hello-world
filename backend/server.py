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
    suggestion_type: str = "general"  # general, bugfix, feature, security, performance, documentation, refactor

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
        
        # Assign to global variables
        qdrant_client = qc
        Distance = Dist
        VectorParams = VP
        PointStruct = PS
        
        # Initialize Qdrant client (in-memory mode)
        qdrant_client_instance = qdrant_client.QdrantClient(":memory:")
        logger.info("Connected to Qdrant (in-memory)")
        
        # Use a simple embedding approach for now
        try:
            from sentence_transformers import SentenceTransformer as ST
            SentenceTransformer = ST
            embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer: {e}")
            # Fallback to simple text hashing for demo
            embedding_model = None
            logger.info("Using simple text embedding fallback")
        
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

def simple_text_embedding(text: str) -> List[float]:
    """Simple fallback embedding using text hashing"""
    import hashlib
    import numpy as np
    
    # Create a hash of the text
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Convert hash to numbers and create a 384-dimensional vector
    hash_numbers = [ord(c) for c in text_hash]
    
    # Pad or truncate to 384 dimensions
    vector = hash_numbers * (384 // len(hash_numbers) + 1)
    vector = vector[:384]
    
    # Normalize the vector
    vector = np.array(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector.tolist()

async def generate_embedding(text: str) -> List[float]:
    """Generate embedding using available method"""
    if embedding_model is not None:
        try:
            return embedding_model.encode(text).tolist()
        except Exception as e:
            logger.warning(f"Embedding model error: {e}, falling back to simple embedding")
    
    # Fallback to simple embedding
    return simple_text_embedding(text)

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
                    embedding = await generate_embedding(embed_text)
                    
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
        query_embedding = await generate_embedding(query)
        
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
        
        # Create enhanced prompts based on suggestion type
        prompt = create_enhanced_prompt(query, suggestion_type, context_chunks)
        
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

def create_enhanced_prompt(query: str, suggestion_type: str, context_chunks: List[Dict]) -> str:
    """Create enhanced prompts for different suggestion types"""
    context_text = format_context_for_prompt(context_chunks)
    
    base_context = f"""You are an expert Ansible and Python developer with deep knowledge of infrastructure automation, configuration management, and DevOps best practices.

Relevant Code Context:
{context_text}

Query: {query}
"""
    
    if suggestion_type == "bugfix":
        return f"""{base_context}

Task: Analyze the code for potential bugs and provide fixes.

Please provide:
1. **Bug Analysis**: Identify the specific issue and root cause
2. **Code Fix**: Provide corrected code with explanations
3. **Testing Strategy**: How to verify the fix works
4. **Prevention**: Best practices to avoid similar issues

Focus on common Ansible and Python pitfalls like variable scoping, task dependencies, error handling, and type issues.

Response:"""

    elif suggestion_type == "feature":
        return f"""{base_context}

Task: Help implement a new feature following existing code patterns.

Please provide:
1. **Implementation Plan**: Step-by-step approach
2. **Code Examples**: Following existing patterns and conventions
3. **Integration Points**: How it fits with existing code
4. **Testing Considerations**: Unit and integration test suggestions

Ensure the solution is maintainable, follows Ansible best practices, and integrates well.

Response:"""

    elif suggestion_type == "security":
        return f"""{base_context}

Task: Analyze code for security vulnerabilities and provide hardening suggestions.

Please provide:
1. **Security Analysis**: Identify potential vulnerabilities
2. **Hardening Recommendations**: Specific security improvements
3. **Secure Code Examples**: Demonstrate secure patterns
4. **Compliance Notes**: Industry standard compliance considerations

Focus on common security issues like credential exposure, privilege escalation, input validation, and secure communications.

Response:"""

    elif suggestion_type == "performance":
        return f"""{base_context}

Task: Analyze code for performance optimization opportunities.

Please provide:
1. **Performance Analysis**: Identify bottlenecks and inefficiencies
2. **Optimization Strategies**: Specific improvements with rationale
3. **Optimized Code**: Demonstrate better performing alternatives
4. **Monitoring Suggestions**: How to measure performance improvements

Focus on Ansible task efficiency, Python code optimization, and infrastructure resource usage.

Response:"""

    elif suggestion_type == "documentation":
        return f"""{base_context}

Task: Generate comprehensive documentation for the code.

Please provide:
1. **Code Documentation**: Clear comments and docstrings
2. **Usage Examples**: How to use the code effectively
3. **Configuration Guide**: Parameter explanations and examples
4. **Troubleshooting**: Common issues and solutions

Follow documentation best practices for both Ansible and Python.

Response:"""

    elif suggestion_type == "refactor":
        return f"""{base_context}

Task: Suggest code refactoring improvements for maintainability and readability.

Please provide:
1. **Refactoring Analysis**: Areas for improvement
2. **Improved Code Structure**: Better organized, more maintainable code
3. **Design Patterns**: Applicable patterns for better architecture
4. **Migration Strategy**: How to safely implement changes

Focus on code organization, reusability, maintainability, and following best practices.

Response:"""

    else:  # general
        return f"""{base_context}

Task: Provide helpful code suggestions and explanations.

Please provide relevant code suggestions, explanations, and best practices for Ansible roles and Python modules.

Response:"""

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

@app.post("/api/analyze-code-quality")
async def analyze_code_quality():
    """Analyze overall code quality metrics"""
    try:
        if processing_status.status != "completed":
            raise HTTPException(status_code=400, detail="Repository not processed yet")
        
        # Get all chunks for analysis
        search_results = qdrant_client_instance.scroll(
            collection_name="code_chunks",
            limit=1000
        )
        
        analysis = {
            "total_files": 0,
            "total_functions": 0,
            "total_classes": 0,
            "total_tasks": 0,
            "complexity_analysis": {},
            "recommendations": []
        }
        
        files_seen = set()
        
        for point in search_results[0]:
            chunk_type = point.payload.get("chunk_type", "unknown")
            file_path = point.payload.get("file_path", "")
            
            if file_path and file_path not in files_seen:
                files_seen.add(file_path)
                analysis["total_files"] += 1
            
            if chunk_type == "python_function":
                analysis["total_functions"] += 1
            elif chunk_type == "python_class":
                analysis["total_classes"] += 1
            elif chunk_type == "ansible_task":
                analysis["total_tasks"] += 1
        
        # Add basic recommendations
        if analysis["total_functions"] > 50:
            analysis["recommendations"].append("Consider breaking down large modules into smaller, focused modules")
        
        if analysis["total_tasks"] > 100:
            analysis["recommendations"].append("Large number of tasks detected - consider role decomposition")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Code quality analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search-similar-code")
async def search_similar_code(request: dict):
    """Find similar code patterns"""
    try:
        if processing_status.status != "completed":
            raise HTTPException(status_code=400, detail="Repository not processed yet")
        
        code_snippet = request.get("code_snippet", "")
        if not code_snippet:
            raise HTTPException(status_code=400, detail="Code snippet required")
        
        # Generate embedding for the code snippet
        snippet_embedding = embedding_model.encode(code_snippet).tolist()
        
        # Search for similar code
        search_results = qdrant_client_instance.search(
            collection_name="code_chunks",
            query_vector=snippet_embedding,
            limit=10,
            score_threshold=0.7  # Only return highly similar results
        )
        
        similar_patterns = []
        for result in search_results:
            similar_patterns.append({
                "file_path": result.payload["file_path"],
                "chunk_type": result.payload["chunk_type"],
                "content": result.payload["content"],
                "similarity_score": result.score,
                "metadata": result.payload["metadata"]
            })
        
        return {
            "query_snippet": code_snippet,
            "similar_patterns": similar_patterns,
            "total_found": len(similar_patterns)
        }
        
    except Exception as e:
        logger.error(f"Similar code search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
