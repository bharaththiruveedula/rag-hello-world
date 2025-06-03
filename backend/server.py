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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Third-party imports for RAG
import tiktoken

# JIRA imports
from jira import JIRA
from jira.exceptions import JIRAError

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

# Configuration from environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "codellama:7b")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# GitLab Configuration
GITLAB_URL = os.getenv("GITLAB_URL", "https://gitlab.com")
GITLAB_API_TOKEN = os.getenv("GITLAB_API_TOKEN")
GITLAB_REPOSITORY_PATH = os.getenv("GITLAB_REPOSITORY_PATH")
GITLAB_BRANCH = os.getenv("GITLAB_BRANCH", "main")

# JIRA Configuration
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_USERNAME = os.getenv("JIRA_USERNAME")  # Optional - for basic auth fallback
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

# Global clients
qdrant_client_instance = None
embedding_model = None
jira_client = None

# Pydantic models
class GitLabConfig(BaseModel):
    gitlab_url: str = "https://gitlab.com"
    api_token: str
    repository_path: str  # e.g., "username/repo-name"
    branch: str = "main"

class RepositoryProcessRequest(BaseModel):
    force_reprocess: bool = False  # Option to force reprocess even if already processed

class JiraTicketRequest(BaseModel):
    ticket_id: str  # e.g., "PROJ-123"
    suggestion_type: str = "general"  # general, bugfix, feature, security, performance, documentation, refactor

class JiraTicketInfo(BaseModel):
    ticket_id: str
    title: str
    description: str
    issue_type: str
    status: str
    assignee: Optional[str] = None
    reporter: Optional[str] = None

class CodeSuggestionRequest(BaseModel):
    query: str
    context: Optional[str] = None
    suggestion_type: str = "general"  # general, bugfix, feature, security, performance, documentation, refactor

class CodeSuggestionResponse(BaseModel):
    suggestion: str
    context_chunks: List[Dict[str, Any]]
    jira_ticket: Optional[JiraTicketInfo] = None
    query: str
    suggestion_type: str
    status: str = "success"

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
    global qdrant_client_instance, embedding_model, jira_client
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
        
        # Initialize JIRA client
        if JIRA_BASE_URL and JIRA_API_TOKEN:
            try:
                # Try token authentication first (recommended for enterprise)
                jira_client = JIRA(
                    server=JIRA_BASE_URL,
                    token_auth=JIRA_API_TOKEN,
                    options={'verify': True}
                )
                logger.info("JIRA client initialized with token authentication")
            except Exception as e:
                # Fallback to basic auth if username is provided
                if JIRA_USERNAME:
                    try:
                        jira_client = JIRA(
                            server=JIRA_BASE_URL,
                            basic_auth=(JIRA_USERNAME, JIRA_API_TOKEN),
                            options={'verify': True}
                        )
                        logger.info("JIRA client initialized with basic authentication (fallback)")
                    except Exception as e2:
                        logger.warning(f"JIRA client initialization failed with both token and basic auth: {e2}")
                        jira_client = None
                else:
                    logger.warning(f"JIRA client initialization with token auth failed: {e}")
                    jira_client = None
        else:
            logger.warning("JIRA credentials not configured")
            jira_client = None
        
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

async def test_jira_connection() -> Dict[str, Any]:
    """Test JIRA connection using JIRA client"""
    global jira_client
    
    if not JIRA_BASE_URL or not JIRA_API_TOKEN:
        return {"status": "error", "message": "JIRA credentials not configured in environment"}
    
    # If jira_client is not initialized, try to initialize it
    if jira_client is None:
        try:
            # Try token authentication first (recommended for enterprise)
            jira_client = JIRA(
                server=JIRA_BASE_URL,
                token_auth=JIRA_API_TOKEN,
                options={'verify': True}
            )
            logger.info("JIRA client initialized with token authentication during test")
        except Exception as e:
            # Fallback to basic auth if username is provided
            if JIRA_USERNAME:
                try:
                    jira_client = JIRA(
                        server=JIRA_BASE_URL,
                        basic_auth=(JIRA_USERNAME, JIRA_API_TOKEN),
                        options={'verify': True}
                    )
                    logger.info("JIRA client initialized with basic authentication during test")
                except Exception as e2:
                    return {"status": "error", "message": f"JIRA client initialization failed: {str(e2)}"}
            else:
                return {"status": "error", "message": f"JIRA token authentication failed: {str(e)}"}
    
    try:
        # Test the connection by getting current user information
        current_user = jira_client.current_user()
        
        return {
            "status": "connected",
            "user": current_user.get("displayName", "Unknown"),
            "account_id": current_user.get("accountId", "Unknown"),
            "auth_method": "token_auth" if not JIRA_USERNAME else "basic_auth"
        }
        
    except JIRAError as e:
        error_msg = f"JIRA API error: {e.status_code} - {e.text}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}
    except Exception as e:
        error_msg = f"JIRA connection error: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}

async def fetch_jira_ticket(ticket_id: str) -> Optional[JiraTicketInfo]:
    """Fetch JIRA ticket information using JIRA client"""
    global jira_client
    
    if not JIRA_BASE_URL or not JIRA_API_TOKEN:
        raise HTTPException(status_code=500, detail="JIRA credentials not configured")
    
    # If jira_client is not initialized, try to initialize it
    if jira_client is None:
        try:
            # Try token authentication first (recommended for enterprise)
            jira_client = JIRA(
                server=JIRA_BASE_URL,
                token_auth=JIRA_API_TOKEN,
                options={'verify': True}
            )
            logger.info("JIRA client initialized with token authentication for ticket fetch")
        except Exception as e:
            # Fallback to basic auth if username is provided
            if JIRA_USERNAME:
                try:
                    jira_client = JIRA(
                        server=JIRA_BASE_URL,
                        basic_auth=(JIRA_USERNAME, JIRA_API_TOKEN),
                        options={'verify': True}
                    )
                    logger.info("JIRA client initialized with basic authentication for ticket fetch")
                except Exception as e2:
                    raise HTTPException(status_code=500, detail=f"JIRA client initialization failed: {str(e2)}")
            else:
                raise HTTPException(status_code=500, detail=f"JIRA token authentication failed: {str(e)}")
    
    try:
        # Fetch the issue using JIRA client
        issue = jira_client.issue(ticket_id, fields='summary,description,issuetype,status,assignee,reporter')
        
        # Extract description text - handle different formats
        description = ""
        if hasattr(issue.fields, 'description') and issue.fields.description:
            if hasattr(issue.fields.description, 'content'):
                # New Atlassian Document Format (ADF)
                try:
                    for content_block in issue.fields.description.content:
                        if content_block.get('type') == 'paragraph':
                            for text_block in content_block.get('content', []):
                                if text_block.get('type') == 'text':
                                    description += text_block.get('text', '') + " "
                except:
                    description = str(issue.fields.description)
            else:
                # Plain text or legacy format
                description = str(issue.fields.description)
        
        return JiraTicketInfo(
            ticket_id=ticket_id,
            title=issue.fields.summary or "",
            description=description.strip(),
            issue_type=issue.fields.issuetype.name if issue.fields.issuetype else "",
            status=issue.fields.status.name if issue.fields.status else "",
            assignee=issue.fields.assignee.displayName if issue.fields.assignee else None,
            reporter=issue.fields.reporter.displayName if issue.fields.reporter else None
        )
        
    except JIRAError as e:
        if e.status_code == 404:
            raise HTTPException(status_code=404, detail=f"JIRA ticket {ticket_id} not found")
        elif e.status_code == 401:
            raise HTTPException(status_code=401, detail="JIRA authentication failed")
        elif e.status_code == 403:
            raise HTTPException(status_code=403, detail=f"Access denied to JIRA ticket {ticket_id}")
        else:
            raise HTTPException(status_code=400, detail=f"JIRA API error: {e.status_code} - {e.text}")
    except Exception as e:
        logger.error(f"Error fetching JIRA ticket {ticket_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch JIRA ticket: {str(e)}")

async def test_gitlab_connection() -> Dict[str, Any]:
    """Test GitLab connection using environment configuration"""
    if not all([GITLAB_URL, GITLAB_API_TOKEN, GITLAB_REPOSITORY_PATH]):
        return {"status": "error", "message": "GitLab credentials not configured in environment"}
    
    try:
        headers = {"Authorization": f"Bearer {GITLAB_API_TOKEN}"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test API access
            response = await client.get(f"{GITLAB_URL}/api/v4/user", headers=headers)
            if response.status_code == 200:
                user_info = response.json()
                
                # Test repository access
                repo_response = await client.get(
                    f"{GITLAB_URL}/api/v4/projects/{GITLAB_REPOSITORY_PATH.replace('/', '%2F')}",
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

async def fetch_repository_contents() -> List[Dict[str, Any]]:
    """Fetch repository contents from GitLab using environment configuration"""
    if not all([GITLAB_URL, GITLAB_API_TOKEN, GITLAB_REPOSITORY_PATH]):
        raise HTTPException(status_code=500, detail="GitLab configuration not complete in environment")
    
    headers = {"Authorization": f"Bearer {GITLAB_API_TOKEN}"}
    all_files = []
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Get repository tree
        url = f"{GITLAB_URL}/api/v4/projects/{GITLAB_REPOSITORY_PATH.replace('/', '%2F')}/repository/tree"
        params = {"ref": GITLAB_BRANCH, "recursive": "true", "per_page": "100"}
        
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
                file_url = f"{GITLAB_URL}/api/v4/projects/{GITLAB_REPOSITORY_PATH.replace('/', '%2F')}/repository/files/{file_info['path'].replace('/', '%2F')}"
                file_params = {"ref": GITLAB_BRANCH}
                
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

async def process_repository_background():
    """Background task to process repository using environment configuration"""
    global processing_status
    
    try:
        processing_status.status = "fetching"
        processing_status.message = "Fetching repository contents..."
        
        # Fetch repository contents
        files = await fetch_repository_contents()
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

async def get_jira_code_suggestions(ticket_id: str, suggestion_type: str = "general") -> CodeSuggestionResponse:
    """Get code suggestions based on JIRA ticket"""
    try:
        # Fetch JIRA ticket information
        jira_ticket = await fetch_jira_ticket(ticket_id)
        
        # Create comprehensive query from JIRA ticket
        query_parts = []
        if jira_ticket.title:
            query_parts.append(f"Title: {jira_ticket.title}")
        if jira_ticket.description:
            query_parts.append(f"Description: {jira_ticket.description}")
        if jira_ticket.issue_type:
            query_parts.append(f"Type: {jira_ticket.issue_type}")
        
        query = "\n".join(query_parts)
        
        # Check if we have any data first
        if qdrant_client_instance is None:
            raise HTTPException(status_code=500, detail="Vector database not initialized")
        
        # Check if collection exists and has data
        try:
            collection_info = qdrant_client_instance.get_collection("code_chunks")
            if collection_info.points_count == 0:
                return CodeSuggestionResponse(
                    suggestion=f"""## JIRA Ticket Analysis: {jira_ticket.ticket_id}

**Issue**: {jira_ticket.title}
**Type**: {jira_ticket.issue_type}
**Status**: {jira_ticket.status}

I've analyzed your JIRA ticket but don't have any repository code processed yet to provide specific suggestions based on your codebase.

**For {suggestion_type} assistance with this ticket:**
1. First, ensure your repository is processed 
2. Once your code is analyzed, I can provide specific suggestions based on your actual Ansible roles and Python modules
3. I'll be able to reference relevant code patterns for this specific issue

**General recommendations for "{jira_ticket.title}":**
- Review existing similar implementations in your codebase
- Consider security implications if this involves infrastructure changes
- Plan for testing and rollback procedures
- Document any configuration changes needed

To get AI-powered code suggestions specific to your codebase, please ensure repository processing is completed.""",
                    context_chunks=[],
                    jira_ticket=jira_ticket,
                    query=query,
                    suggestion_type=suggestion_type,
                    status="no_repository_processed"
                )
        except Exception as e:
            logger.warning(f"Collection check failed: {e}")
        
        # Get regular code suggestions with JIRA context
        suggestions_result = await get_code_suggestions(query, suggestion_type)
        
        # Enhance the suggestion with JIRA context
        enhanced_suggestion = f"""## JIRA Ticket Analysis: {jira_ticket.ticket_id}

**Issue**: {jira_ticket.title}
**Type**: {jira_ticket.issue_type}  
**Status**: {jira_ticket.status}
**Assigned**: {jira_ticket.assignee or "Unassigned"}

---

{suggestions_result['suggestion']}

---

**Next Steps for Ticket {jira_ticket.ticket_id}:**
1. Review the suggested code implementations above
2. Test changes in a development environment
3. Update the JIRA ticket with implementation progress
4. Consider adding automated tests for the changes
5. Plan deployment and rollback strategies
"""

        return CodeSuggestionResponse(
            suggestion=enhanced_suggestion,
            context_chunks=suggestions_result.get('context_chunks', []),
            jira_ticket=jira_ticket,
            query=query,
            suggestion_type=suggestion_type,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"JIRA code suggestion error: {e}")
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
async def test_connections():
    """Test all external connections using environment configuration"""
    results = {}
    
    # Test OLLAMA
    results["ollama"] = await test_ollama_connection()
    
    # Test GitLab  
    results["gitlab"] = await test_gitlab_connection()
    
    # Test JIRA
    results["jira"] = await test_jira_connection()
    
    # Test Qdrant
    try:
        collections = qdrant_client_instance.get_collections()
        results["qdrant"] = {"status": "connected", "collections": len(collections.collections)}
    except Exception as e:
        results["qdrant"] = {"status": "error", "message": str(e)}
    
    return results

@app.post("/api/process-repository")
async def process_repository(request: RepositoryProcessRequest, background_tasks: BackgroundTasks):
    """Start processing repository using environment configuration"""
    global processing_status
    
    if processing_status.status == "processing" and not request.force_reprocess:
        raise HTTPException(status_code=400, detail="Repository processing already in progress")
    
    processing_status = ProcessingStatus(status="starting", message="Initializing repository processing...")
    background_tasks.add_task(process_repository_background)
    
    return {"message": "Repository processing started", "status": processing_status}

@app.post("/api/jira-suggest")
async def jira_suggest(request: JiraTicketRequest):
    """Get code suggestions based on JIRA ticket"""
    try:
        result = await get_jira_code_suggestions(request.ticket_id, request.suggestion_type)
        return result
    except Exception as e:
        logger.error(f"JIRA suggestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/suggest")
async def suggest_code(request: CodeSuggestionRequest):
    """Get code suggestions"""
    try:
        # Check if we have any data first
        if qdrant_client_instance is None:
            raise HTTPException(status_code=500, detail="Vector database not initialized")
        
        # Check if collection exists and has data
        try:
            collection_info = qdrant_client_instance.get_collection("code_chunks")
            if collection_info.points_count == 0:
                # Return a helpful message instead of error for better UX
                return {
                    "suggestion": f"""I understand you're asking about: "{request.query}"

However, I don't have any repository code processed yet to provide specific suggestions based on your codebase.

Here are some general recommendations for your query type ({request.suggestion_type}):

**For {request.suggestion_type} assistance:**
1. First, process a repository using the Setup tab
2. Once your code is analyzed, I can provide specific suggestions based on your actual codebase patterns
3. I'll be able to reference your existing Ansible roles and Python modules for contextual help

To get started:
- Go to the Setup tab
- Enter your GitLab credentials and repository details  
- Click "Process Repository"
- Return here for AI-powered code suggestions!""",
                    "context_chunks": [],
                    "query": request.query,
                    "suggestion_type": request.suggestion_type,
                    "status": "no_repository_processed"
                }
        except Exception as e:
            logger.warning(f"Collection check failed: {e}")
            # Create collection if it doesn't exist
            try:
                qdrant_client_instance.create_collection(
                    collection_name="code_chunks",
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
            except Exception as create_error:
                logger.error(f"Failed to create collection: {create_error}")
        
        result = await get_code_suggestions(request.query, request.suggestion_type)
        return result
        
    except Exception as e:
        logger.error(f"Suggestion endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Suggestion generation failed: {str(e)}")

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
        if qdrant_client_instance is None:
            raise HTTPException(status_code=500, detail="Vector database not initialized")
        
        # Check if collection exists and has data
        try:
            collection_info = qdrant_client_instance.get_collection("code_chunks")
            if collection_info.points_count == 0:
                return {
                    "message": "No repository processed yet",
                    "total_files": 0,
                    "total_functions": 0,
                    "total_classes": 0,
                    "total_tasks": 0,
                    "complexity_analysis": {},
                    "recommendations": [
                        "Process a repository first to get code quality analysis",
                        "Go to Setup tab and configure GitLab repository",
                        "Click 'Process Repository' to analyze your code"
                    ]
                }
        except Exception as e:
            logger.warning(f"Collection check failed: {e}")
            return {
                "error": f"Could not access code data: {str(e)}",
                "total_files": 0,
                "total_functions": 0,
                "total_classes": 0,
                "total_tasks": 0,
                "recommendations": ["Please process a repository first"]
            }
        
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
            
        if analysis["total_files"] == 0:
            analysis["recommendations"].append("No code files detected - ensure repository contains Ansible roles or Python modules")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Code quality analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search-similar-code")
async def search_similar_code(request: dict):
    """Find similar code patterns"""
    try:
        code_snippet = request.get("code_snippet", "")
        if not code_snippet:
            raise HTTPException(status_code=400, detail="Code snippet required")
        
        if qdrant_client_instance is None:
            raise HTTPException(status_code=500, detail="Vector database not initialized")
        
        # Check if collection exists and has data
        try:
            collection_info = qdrant_client_instance.get_collection("code_chunks")
            if collection_info.points_count == 0:
                return {
                    "query_snippet": code_snippet,
                    "similar_patterns": [],
                    "total_found": 0,
                    "message": "No repository processed yet. Process a repository first to search for similar code patterns."
                }
        except Exception as e:
            logger.warning(f"Collection check failed: {e}")
            return {
                "query_snippet": code_snippet,
                "similar_patterns": [],
                "total_found": 0,
                "error": f"Could not access code data: {str(e)}"
            }
        
        # Generate embedding for the code snippet
        snippet_embedding = await generate_embedding(code_snippet)
        
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
