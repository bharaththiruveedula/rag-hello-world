import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function App() {
  const [activeTab, setActiveTab] = useState('process');
  const [connectionStatus, setConnectionStatus] = useState(null);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [repositoryInfo, setRepositoryInfo] = useState(null);
  const [jiraTicketId, setJiraTicketId] = useState('');
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
        headers: { 'Content-Type': 'application/json' }
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
        body: JSON.stringify({ force_reprocess: false })
      });
      const result = await response.json();
      console.log('Processing started:', result);
      setActiveTab('status');
    } catch (error) {
      console.error('Repository processing error:', error);
    }
    setLoading(false);
  };

  const getJiraSuggestions = async () => {
    if (!jiraTicketId.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/jira-suggest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticket_id: jiraTicketId,
          suggestion_type: suggestionType
        })
      });
      const result = await response.json();
      setSuggestions(result);
    } catch (error) {
      console.error('JIRA suggestion error:', error);
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
      <div className="glass-card mt-8">
        <div className="flex items-center gap-3 mb-6">
          <div className="connection-icon">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
            </svg>
          </div>
          <h3 className="text-xl font-bold gradient-text">System Connections</h3>
        </div>
        
        <div className="grid gap-4">
          {/* OLLAMA Status */}
          <div className={`status-pill ${connectionStatus.ollama?.status === 'connected' ? 'status-success' : 'status-error'}`}>
            <div className="status-indicator">
              <div className="status-dot"></div>
              <span className="status-icon">🤖</span>
              <span className="font-semibold">OLLAMA</span>
            </div>
            <div className="status-badge">
              {connectionStatus.ollama?.status || 'error'}
            </div>
            {connectionStatus.ollama?.models && (
              <div className="status-details">
                Models: {connectionStatus.ollama.models.join(', ')}
              </div>
            )}
            {connectionStatus.ollama?.message && (
              <div className="status-error-msg">{connectionStatus.ollama.message}</div>
            )}
          </div>

          {/* GitLab Status */}
          <div className={`status-pill ${connectionStatus.gitlab?.status === 'connected' ? 'status-success' : 'status-error'}`}>
            <div className="status-indicator">
              <div className="status-dot"></div>
              <span className="status-icon">🦊</span>
              <span className="font-semibold">GitLab</span>
            </div>
            <div className="status-badge">
              {connectionStatus.gitlab?.status || 'error'}
            </div>
            {connectionStatus.gitlab?.user && (
              <div className="status-details">
                User: {connectionStatus.gitlab.user}, Repository: {connectionStatus.gitlab.repository}
              </div>
            )}
            {connectionStatus.gitlab?.message && (
              <div className="status-error-msg">{connectionStatus.gitlab.message}</div>
            )}
          </div>

          {/* JIRA Status */}
          <div className={`status-pill ${connectionStatus.jira?.status === 'connected' ? 'status-success' : 'status-error'}`}>
            <div className="status-indicator">
              <div className="status-dot"></div>
              <span className="status-icon">🎫</span>
              <span className="font-semibold">JIRA</span>
            </div>
            <div className="status-badge">
              {connectionStatus.jira?.status || 'error'}
            </div>
            {connectionStatus.jira?.user && (
              <div className="status-details">
                User: {connectionStatus.jira.user}, Account: {connectionStatus.jira.account_id}
              </div>
            )}
            {connectionStatus.jira?.message && (
              <div className="status-error-msg">{connectionStatus.jira.message}</div>
            )}
          </div>

          {/* Qdrant Status */}
          <div className={`status-pill ${connectionStatus.qdrant?.status === 'connected' ? 'status-success' : 'status-error'}`}>
            <div className="status-indicator">
              <div className="status-dot"></div>
              <span className="status-icon">🔍</span>
              <span className="font-semibold">Vector DB</span>
            </div>
            <div className="status-badge">
              {connectionStatus.qdrant?.status || 'error'}
            </div>
            {connectionStatus.qdrant?.collections !== undefined && (
              <div className="status-details">
                Collections: {connectionStatus.qdrant.collections}
              </div>
            )}
            {connectionStatus.qdrant?.message && (
              <div className="status-error-msg">{connectionStatus.qdrant.message}</div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderProcessingStatus = () => {
    if (!processingStatus) return null;

    const getStatusColor = (status) => {
      switch (status) {
        case 'completed': return 'text-emerald-400';
        case 'error': return 'text-red-400';
        case 'idle': return 'text-slate-400';
        default: return 'text-blue-400';
      }
    };

    return (
      <div className="glass-card">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="processing-icon">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold gradient-text">Repository Processing</h3>
          </div>
          <div className={`status-badge ${getStatusColor(processingStatus.status)}`}>
            {processingStatus.status}
          </div>
        </div>
        
        <div className="status-message mb-4">
          {processingStatus.message}
        </div>
        
        {processingStatus.total_files > 0 && (
          <div className="progress-container">
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
      <div className="glass-card mt-8">
        <div className="flex items-center gap-3 mb-6">
          <div className="repo-icon">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h3 className="text-xl font-bold gradient-text">Repository Analysis</h3>
        </div>
        
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-value">{repositoryInfo.total_chunks}</div>
            <div className="metric-label">Total Code Chunks</div>
          </div>
          
          <div className="metric-card">
            <div className="metric-label mb-3">Chunk Types</div>
            <div className="space-y-2">
              {Object.entries(repositoryInfo.chunk_types || {}).map(([type, count]) => (
                <div key={type} className="flex justify-between items-center">
                  <span className="text-sm text-slate-300">{type.replace('_', ' ')}</span>
                  <span className="metric-badge">{count}</span>
                </div>
              ))}
            </div>
          </div>
          
          <div className="metric-card">
            <div className="metric-label mb-3">File Types</div>
            <div className="space-y-2">
              {Object.entries(repositoryInfo.file_types || {}).map(([type, count]) => (
                <div key={type} className="flex justify-between items-center">
                  <span className="text-sm text-slate-300">{type}</span>
                  <span className="metric-badge">{count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderJiraSuggestions = () => {
    if (!suggestions) return null;

    return (
      <div className="glass-card mt-8">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="suggestion-icon">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold gradient-text">AI Code Suggestion</h3>
          </div>
          <div className="suggestion-type-badge">{suggestionType}</div>
        </div>
        
        {suggestions.jira_ticket && (
          <div className="jira-ticket-card mb-6">
            <div className="ticket-header">
              <span className="ticket-id">{suggestions.jira_ticket.ticket_id}</span>
              <span className="ticket-status">{suggestions.jira_ticket.status}</span>
            </div>
            <h4 className="ticket-title">{suggestions.jira_ticket.title}</h4>
            <div className="ticket-meta">
              <span>Type: {suggestions.jira_ticket.issue_type}</span>
              {suggestions.jira_ticket.assignee && (
                <span>Assignee: {suggestions.jira_ticket.assignee}</span>
              )}
            </div>
          </div>
        )}
        
        <div className="suggestion-content">
          <div className="suggestion-text">
            {suggestions.suggestion.split('\n').map((line, index) => {
              let lineClass = 'text-line';
              if (line.startsWith('```')) {
                lineClass = 'code-block-marker';
              } else if (line.startsWith('+')) {
                lineClass = 'diff-added';
              } else if (line.startsWith('-')) {
                lineClass = 'diff-removed';
              } else if (line.startsWith('@@')) {
                lineClass = 'diff-hunk';
              } else if (
                line.startsWith('diff') ||
                line.startsWith('index') ||
                line.startsWith('---') ||
                line.startsWith('+++')
              ) {
                lineClass = 'diff-meta';
              }
              return (
                <div key={index} className={lineClass}>
                  {line}
                </div>
              );
            })}
          </div>
        </div>
        
        {suggestions.context_chunks && suggestions.context_chunks.length > 0 && (
          <div className="context-chunks mt-6">
            <h4 className="context-header">Referenced Code Context</h4>
            <div className="space-y-4">
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
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="app-container">
      {/* Animated Background */}
      <div className="bg-animation">
        <div className="bg-gradient-1"></div>
        <div className="bg-gradient-2"></div>
        <div className="bg-gradient-3"></div>
      </div>

      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">
              <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <div>
              <h1 className="app-title">RAG Code Assistant</h1>
              <p className="app-subtitle">Enterprise JIRA-integrated AI for Infrastructure Automation</p>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="nav-container">
        <div className="nav-content">
          {[
            { id: 'process', label: 'Process', icon: '⚡' },
            { id: 'status', label: 'Status', icon: '📊' },
            { id: 'analytics', label: 'Analytics', icon: '🔬' },
            { id: 'jira', label: 'JIRA AI', icon: '🎫' }
          ].map((tab) => (
            <button
              key={tab.id}
              className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span className="nav-icon">{tab.icon}</span>
              <span className="nav-label">{tab.label}</span>
            </button>
          ))}
        </div>
      </nav>

      {/* Main Content */}
      <main className="main-content">
        {activeTab === 'process' && (
          <div className="content-section">
            <div className="glass-card">
              <div className="flex items-center gap-3 mb-6">
                <div className="section-icon">
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <div>
                  <h2 className="section-title">Repository Processing</h2>
                  <p className="section-description">
                    Process your GitLab repository to enable AI-powered code suggestions. 
                    All configuration is managed via secure environment variables.
                  </p>
                </div>
              </div>
              
              <div className="action-buttons">
                <button 
                  className="btn btn-secondary"
                  onClick={testConnections}
                  disabled={loading}
                >
                  <svg className="btn-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
                  </svg>
                  {loading ? 'Testing Connections...' : 'Test Connections'}
                </button>
                
                <button 
                  className="btn btn-primary"
                  onClick={processRepository}
                  disabled={loading}
                >
                  <svg className="btn-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  {loading ? 'Processing Repository...' : 'Process Repository'}
                </button>
              </div>
            </div>
            
            {renderConnectionStatus()}
          </div>
        )}

        {activeTab === 'status' && (
          <div className="content-section">
            {renderProcessingStatus()}
            {repositoryInfo && renderRepositoryInfo()}
          </div>
        )}

        {activeTab === 'analytics' && (
          <div className="content-section">
            <div className="glass-card">
              <div className="flex items-center gap-3 mb-6">
                <div className="section-icon">
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <div>
                  <h2 className="section-title">Code Analytics & Insights</h2>
                  <p className="section-description">
                    Advanced analysis of your codebase patterns and quality metrics.
                  </p>
                </div>
              </div>
              
              {processingStatus?.status !== 'completed' ? (
                <div className="empty-state">
                  <div className="empty-icon">📊</div>
                  <p className="empty-message">Please process a repository first to enable code analytics.</p>
                </div>
              ) : (
                <div className="analytics-grid">
                  {/* Code Quality Analysis */}
                  <div className="analytics-card">
                    <div className="card-header">
                      <h3 className="card-title">Code Quality Analysis</h3>
                      <button 
                        className="btn btn-sm"
                        onClick={analyzeCodeQuality}
                        disabled={loading}
                      >
                        {loading ? 'Analyzing...' : 'Analyze'}
                      </button>
                    </div>
                    
                    {codeQuality && (
                      <div className="quality-results">
                        <div className="metrics-grid">
                          <div className="metric-small">
                            <div className="metric-value">{codeQuality.total_files}</div>
                            <div className="metric-label">Files</div>
                          </div>
                          <div className="metric-small">
                            <div className="metric-value">{codeQuality.total_functions}</div>
                            <div className="metric-label">Functions</div>
                          </div>
                          <div className="metric-small">
                            <div className="metric-value">{codeQuality.total_classes}</div>
                            <div className="metric-label">Classes</div>
                          </div>
                          <div className="metric-small">
                            <div className="metric-value">{codeQuality.total_tasks}</div>
                            <div className="metric-label">Tasks</div>
                          </div>
                        </div>
                        
                        {codeQuality.recommendations.length > 0 && (
                          <div className="recommendations">
                            <h4 className="recommendations-title">Recommendations</h4>
                            <ul className="recommendations-list">
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
                    <div className="card-header">
                      <h3 className="card-title">Similar Code Pattern Search</h3>
                    </div>
                    
                    <div className="input-group">
                      <label className="input-label">Code Snippet</label>
                      <textarea
                        className="modern-textarea"
                        value={similarCodeQuery}
                        onChange={(e) => setSimilarCodeQuery(e.target.value)}
                        placeholder="Paste code snippet to find similar patterns..."
                        rows={6}
                      />
                    </div>
                    
                    <button 
                      className="btn btn-primary w-full"
                      onClick={searchSimilarCode}
                      disabled={loading || !similarCodeQuery.trim()}
                    >
                      {loading ? 'Searching...' : 'Find Similar Code'}
                    </button>
                    
                    {similarCodeResults && (
                      <div className="search-results">
                        <h4 className="results-title">Found {similarCodeResults.total_found} Similar Patterns</h4>
                        <div className="space-y-4">
                          {similarCodeResults.similar_patterns.map((pattern, index) => (
                            <div key={index} className="pattern-result">
                              <div className="pattern-meta">
                                <span className="pattern-file">{pattern.file_path}</span>
                                <span className="pattern-type">{pattern.chunk_type}</span>
                                <span className="pattern-score">
                                  {(pattern.similarity_score * 100).toFixed(1)}%
                                </span>
                              </div>
                              <pre className="pattern-code">{pattern.content}</pre>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'jira' && (
          <div className="content-section">
            <div className="glass-card">
              <div className="flex items-center gap-3 mb-6">
                <div className="section-icon">
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 5v2m0 4v2m0 4v2M5 5a2 2 0 00-2 2v3a2 2 0 110 4v3a2 2 0 002 2h14a2 2 0 002-2v-3a2 2 0 110-4V7a2 2 0 00-2-2H5z" />
                  </svg>
                </div>
                <div>
                  <h2 className="section-title">JIRA AI Assistant</h2>
                  <p className="section-description">
                    Get intelligent code suggestions based on your JIRA tickets and processed repository.
                  </p>
                </div>
              </div>
              
              {processingStatus?.status !== 'completed' ? (
                <div className="empty-state">
                  <div className="empty-icon">🎫</div>
                  <p className="empty-message">Please process a repository first to enable JIRA-based code suggestions.</p>
                </div>
              ) : (
                <div className="jira-interface">
                  <div className="input-grid">
                    <div className="input-group">
                      <label className="input-label">Suggestion Type</label>
                      <select
                        className="modern-select"
                        value={suggestionType}
                        onChange={(e) => setSuggestionType(e.target.value)}
                      >
                        <option value="general">💡 General Help</option>
                        <option value="bugfix">🔧 Bug Fix</option>
                        <option value="feature">✨ New Feature</option>
                        <option value="security">🔒 Security Analysis</option>
                        <option value="performance">⚡ Performance Optimization</option>
                        <option value="documentation">📖 Documentation</option>
                        <option value="refactor">🔄 Code Refactoring</option>
                      </select>
                    </div>
                    
                    <div className="input-group">
                      <label className="input-label">JIRA Ticket ID</label>
                      <input
                        className="modern-input"
                        type="text"
                        value={jiraTicketId}
                        onChange={(e) => setJiraTicketId(e.target.value)}
                        placeholder="e.g., PROJ-123"
                      />
                    </div>
                  </div>
                  
                  <button 
                    className="btn btn-primary btn-large"
                    onClick={getJiraSuggestions}
                    disabled={loading || !jiraTicketId.trim()}
                  >
                    <svg className="btn-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                    {loading ? 'Generating AI Suggestions...' : 'Get JIRA AI Suggestions'}
                  </button>
                  
                  {renderJiraSuggestions()}
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
