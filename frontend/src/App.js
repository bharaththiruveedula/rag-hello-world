import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function App() {
  const [activeTab, setActiveTab] = useState('setup');
  const [gitlabConfig, setGitlabConfig] = useState({
    gitlab_url: 'https://gitlab.com',
    api_token: '',
    repository_path: '',
    branch: 'main'
  });
  const [connectionStatus, setConnectionStatus] = useState(null);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [repositoryInfo, setRepositoryInfo] = useState(null);
  const [query, setQuery] = useState('');
  const [suggestionType, setSuggestionType] = useState('general');
  const [suggestions, setSuggestions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [codeQuality, setCodeQuality] = useState(null);
  const [similarCodeQuery, setSimilarCodeQuery] = useState('');
  const [similarCodeResults, setSimilarCodeResults] = useState(null);

  // Poll for processing status
  useEffect(() => {
    const pollStatus = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/status`);
        const status = await response.json();
        setProcessingStatus(status);
        
        // If completed, fetch repository info
        if (status.status === 'completed' && !repositoryInfo) {
          fetchRepositoryInfo();
        }
      } catch (error) {
        console.error('Status polling error:', error);
      }
    };

    const interval = setInterval(pollStatus, 2000);
    return () => clearInterval(interval);
  }, [repositoryInfo]);

  const fetchRepositoryInfo = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/repository-info`);
      const info = await response.json();
      setRepositoryInfo(info);
    } catch (error) {
      console.error('Repository info error:', error);
    }
  };

  const testConnections = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/test-connections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(gitlabConfig)
      });
      const results = await response.json();
      setConnectionStatus(results);
    } catch (error) {
      console.error('Connection test error:', error);
      setConnectionStatus({ error: error.message });
    }
    setLoading(false);
  };

  const processRepository = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/process-repository`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: gitlabConfig })
      });
      const result = await response.json();
      console.log('Processing started:', result);
      setActiveTab('status');
    } catch (error) {
      console.error('Repository processing error:', error);
    }
    setLoading(false);
  };

  const getSuggestions = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/suggest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          suggestion_type: suggestionType
        })
      });
      const result = await response.json();
      setSuggestions(result);
    } catch (error) {
      console.error('Suggestion error:', error);
    }
    setLoading(false);
  };

  const analyzeCodeQuality = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/analyze-code-quality`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const result = await response.json();
      setCodeQuality(result);
    } catch (error) {
      console.error('Code quality analysis error:', error);
    }
    setLoading(false);
  };

  const searchSimilarCode = async () => {
    if (!similarCodeQuery.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/search-similar-code`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code_snippet: similarCodeQuery
        })
      });
      const result = await response.json();
      setSimilarCodeResults(result);
    } catch (error) {
      console.error('Similar code search error:', error);
    }
    setLoading(false);
  };

  const renderConnectionStatus = () => {
    if (!connectionStatus) return null;

    return (
      <div className="connection-status">
        <h3 className="text-lg font-semibold mb-4">Connection Status</h3>
        
        {/* OLLAMA Status */}
        <div className={`status-item ${connectionStatus.ollama?.status === 'connected' ? 'success' : 'error'}`}>
          <div className="status-header">
            <span className="status-icon">🤖</span>
            <span>OLLAMA</span>
            <span className={`status-badge ${connectionStatus.ollama?.status === 'connected' ? 'success' : 'error'}`}>
              {connectionStatus.ollama?.status || 'error'}
            </span>
          </div>
          {connectionStatus.ollama?.models && (
            <div className="status-details">
              Available models: {connectionStatus.ollama.models.join(', ')}
            </div>
          )}
          {connectionStatus.ollama?.message && (
            <div className="status-error">{connectionStatus.ollama.message}</div>
          )}
        </div>

        {/* GitLab Status */}
        <div className={`status-item ${connectionStatus.gitlab?.status === 'connected' ? 'success' : 'error'}`}>
          <div className="status-header">
            <span className="status-icon">🦊</span>
            <span>GitLab</span>
            <span className={`status-badge ${connectionStatus.gitlab?.status === 'connected' ? 'success' : 'error'}`}>
              {connectionStatus.gitlab?.status || 'error'}
            </span>
          </div>
          {connectionStatus.gitlab?.user && (
            <div className="status-details">
              User: {connectionStatus.gitlab.user}, Repository: {connectionStatus.gitlab.repository}
            </div>
          )}
          {connectionStatus.gitlab?.message && (
            <div className="status-error">{connectionStatus.gitlab.message}</div>
          )}
        </div>

        {/* Qdrant Status */}
        <div className={`status-item ${connectionStatus.qdrant?.status === 'connected' ? 'success' : 'error'}`}>
          <div className="status-header">
            <span className="status-icon">🔍</span>
            <span>Qdrant Vector DB</span>
            <span className={`status-badge ${connectionStatus.qdrant?.status === 'connected' ? 'success' : 'error'}`}>
              {connectionStatus.qdrant?.status || 'error'}
            </span>
          </div>
          {connectionStatus.qdrant?.collections !== undefined && (
            <div className="status-details">
              Collections: {connectionStatus.qdrant.collections}
            </div>
          )}
          {connectionStatus.qdrant?.message && (
            <div className="status-error">{connectionStatus.qdrant.message}</div>
          )}
        </div>
      </div>
    );
  };

  const renderProcessingStatus = () => {
    if (!processingStatus) return null;

    const getStatusColor = (status) => {
      switch (status) {
        case 'completed': return 'text-green-600';
        case 'error': return 'text-red-600';
        case 'idle': return 'text-gray-600';
        default: return 'text-blue-600';
      }
    };

    return (
      <div className="processing-status">
        <div className="status-header">
          <h3 className="text-lg font-semibold">Repository Processing</h3>
          <span className={`status-badge ${getStatusColor(processingStatus.status)}`}>
            {processingStatus.status}
          </span>
        </div>
        
        <div className="status-message">
          {processingStatus.message}
        </div>
        
        {processingStatus.total_files > 0 && (
          <div className="progress-info">
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{ 
                  width: `${(processingStatus.processed_files / processingStatus.total_files) * 100}%` 
                }}
              />
            </div>
            <div className="progress-text">
              {processingStatus.processed_files} / {processingStatus.total_files} files processed
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderRepositoryInfo = () => {
    if (!repositoryInfo) return null;

    return (
      <div className="repository-info">
        <h3 className="text-lg font-semibold mb-4">Repository Analysis</h3>
        
        <div className="info-grid">
          <div className="info-card">
            <div className="info-header">
              <span className="info-icon">📊</span>
              <span>Total Code Chunks</span>
            </div>
            <div className="info-value">{repositoryInfo.total_chunks}</div>
          </div>
          
          <div className="info-card">
            <div className="info-header">
              <span className="info-icon">🔧</span>
              <span>Chunk Types</span>
            </div>
            <div className="info-details">
              {Object.entries(repositoryInfo.chunk_types || {}).map(([type, count]) => (
                <div key={type} className="detail-item">
                  <span className="detail-label">{type.replace('_', ' ')}</span>
                  <span className="detail-value">{count}</span>
                </div>
              ))}
            </div>
          </div>
          
          <div className="info-card">
            <div className="info-header">
              <span className="info-icon">📁</span>
              <span>File Types</span>
            </div>
            <div className="info-details">
              {Object.entries(repositoryInfo.file_types || {}).map(([type, count]) => (
                <div key={type} className="detail-item">
                  <span className="detail-label">{type}</span>
                  <span className="detail-value">{count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderSuggestions = () => {
    if (!suggestions) return null;

    return (
      <div className="suggestions-result">
        <div className="suggestion-header">
          <h3 className="text-lg font-semibold">Code Suggestion</h3>
          <span className="suggestion-type">{suggestionType}</span>
        </div>
        
        <div className="suggestion-content">
          <div className="suggestion-text">
            {suggestions.suggestion.split('\n').map((line, index) => (
              <div key={index} className={line.startsWith('```') ? 'code-block' : 'text-line'}>
                {line}
              </div>
            ))}
          </div>
        </div>
        
        {suggestions.context_chunks && suggestions.context_chunks.length > 0 && (
          <div className="context-chunks">
            <h4 className="context-header">Referenced Code Context</h4>
            {suggestions.context_chunks.map((chunk, index) => (
              <div key={index} className="context-item">
                <div className="context-meta">
                  <span className="context-file">{chunk.file_path}</span>
                  <span className="context-type">{chunk.chunk_type}</span>
                  <span className="context-score">Score: {chunk.score.toFixed(3)}</span>
                </div>
                <pre className="context-code">{chunk.content}</pre>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1 className="app-title">RAG Code Assistant</h1>
          <p className="app-subtitle">AI-powered code suggestions for Ansible roles and Python modules</p>
        </div>
      </header>

      <nav className="tab-navigation">
        {['setup', 'status', 'analytics', 'chat'].map((tab) => (
          <button
            key={tab}
            className={`tab-button ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </nav>

      <main className="main-content">
        {activeTab === 'setup' && (
          <div className="setup-section">
            <div className="section-card">
              <h2 className="section-title">GitLab Configuration</h2>
              
              <div className="form-group">
                <label className="form-label">GitLab URL</label>
                <input
                  type="text"
                  className="form-input"
                  value={gitlabConfig.gitlab_url}
                  onChange={(e) => setGitlabConfig({...gitlabConfig, gitlab_url: e.target.value})}
                  placeholder="https://gitlab.com"
                />
              </div>
              
              <div className="form-group">
                <label className="form-label">API Token</label>
                <input
                  type="password"
                  className="form-input"
                  value={gitlabConfig.api_token}
                  onChange={(e) => setGitlabConfig({...gitlabConfig, api_token: e.target.value})}
                  placeholder="your-gitlab-api-token"
                />
              </div>
              
              <div className="form-group">
                <label className="form-label">Repository Path</label>
                <input
                  type="text"
                  className="form-input"
                  value={gitlabConfig.repository_path}
                  onChange={(e) => setGitlabConfig({...gitlabConfig, repository_path: e.target.value})}
                  placeholder="username/repository-name"
                />
              </div>
              
              <div className="form-group">
                <label className="form-label">Branch</label>
                <input
                  type="text"
                  className="form-input"
                  value={gitlabConfig.branch}
                  onChange={(e) => setGitlabConfig({...gitlabConfig, branch: e.target.value})}
                  placeholder="main"
                />
              </div>
              
              <div className="button-group">
                <button 
                  className="btn btn-secondary"
                  onClick={testConnections}
                  disabled={loading}
                >
                  {loading ? 'Testing...' : 'Test Connections'}
                </button>
                
                <button 
                  className="btn btn-primary"
                  onClick={processRepository}
                  disabled={loading || !gitlabConfig.api_token || !gitlabConfig.repository_path}
                >
                  {loading ? 'Processing...' : 'Process Repository'}
                </button>
              </div>
            </div>
            
            {renderConnectionStatus()}
          </div>
        )}

        {activeTab === 'status' && (
          <div className="status-section">
            {renderProcessingStatus()}
            {repositoryInfo && renderRepositoryInfo()}
          </div>
        )}

        {activeTab === 'analytics' && (
          <div className="analytics-section">
            <div className="section-card">
              <h2 className="section-title">Code Analytics & Insights</h2>
              
              {processingStatus?.status !== 'completed' ? (
                <div className="analytics-disabled">
                  <p>Please process a repository first to enable code analytics.</p>
                </div>
              ) : (
                <div className="analytics-interface">
                  {/* Code Quality Analysis */}
                  <div className="analytics-card">
                    <div className="analytics-header">
                      <h3>Code Quality Analysis</h3>
                      <button 
                        className="btn btn-secondary"
                        onClick={analyzeCodeQuality}
                        disabled={loading}
                      >
                        {loading ? 'Analyzing...' : 'Analyze Quality'}
                      </button>
                    </div>
                    
                    {codeQuality && (
                      <div className="quality-results">
                        <div className="quality-metrics">
                          <div className="metric-item">
                            <span className="metric-label">Total Files</span>
                            <span className="metric-value">{codeQuality.total_files}</span>
                          </div>
                          <div className="metric-item">
                            <span className="metric-label">Python Functions</span>
                            <span className="metric-value">{codeQuality.total_functions}</span>
                          </div>
                          <div className="metric-item">
                            <span className="metric-label">Python Classes</span>
                            <span className="metric-value">{codeQuality.total_classes}</span>
                          </div>
                          <div className="metric-item">
                            <span className="metric-label">Ansible Tasks</span>
                            <span className="metric-value">{codeQuality.total_tasks}</span>
                          </div>
                        </div>
                        
                        {codeQuality.recommendations.length > 0 && (
                          <div className="recommendations">
                            <h4>Recommendations</h4>
                            <ul>
                              {codeQuality.recommendations.map((rec, index) => (
                                <li key={index}>{rec}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  
                  {/* Similar Code Search */}
                  <div className="analytics-card">
                    <div className="analytics-header">
                      <h3>Similar Code Pattern Search</h3>
                    </div>
                    
                    <div className="form-group">
                      <label className="form-label">Code Snippet</label>
                      <textarea
                        className="form-textarea"
                        value={similarCodeQuery}
                        onChange={(e) => setSimilarCodeQuery(e.target.value)}
                        placeholder="Paste code snippet to find similar patterns..."
                        rows={6}
                      />
                    </div>
                    
                    <button 
                      className="btn btn-primary"
                      onClick={searchSimilarCode}
                      disabled={loading || !similarCodeQuery.trim()}
                    >
                      {loading ? 'Searching...' : 'Find Similar Code'}
                    </button>
                    
                    {similarCodeResults && (
                      <div className="similar-results">
                        <h4>Found {similarCodeResults.total_found} Similar Patterns</h4>
                        {similarCodeResults.similar_patterns.map((pattern, index) => (
                          <div key={index} className="similar-pattern">
                            <div className="pattern-meta">
                              <span className="pattern-file">{pattern.file_path}</span>
                              <span className="pattern-type">{pattern.chunk_type}</span>
                              <span className="pattern-score">
                                Similarity: {(pattern.similarity_score * 100).toFixed(1)}%
                              </span>
                            </div>
                            <pre className="pattern-code">{pattern.content}</pre>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'chat' && (
          <div className="chat-section">
            <div className="section-card">
              <h2 className="section-title">Code Assistant Chat</h2>
              
              {processingStatus?.status !== 'completed' ? (
                <div className="chat-disabled">
                  <p>Please process a repository first to enable code suggestions.</p>
                </div>
              ) : (
                <div className="chat-interface">
                  <div className="query-section">
                    <div className="form-group">
                      <label className="form-label">Suggestion Type</label>
                      <select
                        className="form-select"
                        value={suggestionType}
                        onChange={(e) => setSuggestionType(e.target.value)}
                      >
                        <option value="general">General Help</option>
                        <option value="bugfix">Bug Fix</option>
                        <option value="feature">New Feature</option>
                        <option value="security">Security Analysis</option>
                        <option value="performance">Performance Optimization</option>
                        <option value="documentation">Documentation</option>
                        <option value="refactor">Code Refactoring</option>
                      </select>
                    </div>
                    
                    <div className="form-group">
                      <label className="form-label">Your Question</label>
                      <textarea
                        className="form-textarea"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Ask about your Ansible roles or Python modules..."
                        rows={4}
                      />
                    </div>
                    
                    <button 
                      className="btn btn-primary"
                      onClick={getSuggestions}
                      disabled={loading || !query.trim()}
                    >
                      {loading ? 'Getting Suggestions...' : 'Get Suggestions'}
                    </button>
                  </div>
                  
                  {renderSuggestions()}
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
