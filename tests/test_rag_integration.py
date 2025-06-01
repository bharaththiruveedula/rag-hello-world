#!/usr/bin/env python3
"""
Comprehensive test suite for RAG Code Assistant with real integrations.
Tests GitLab API, OLLAMA, and full end-to-end workflows.
"""

import os
import sys
import asyncio
import httpx
import json
import time
from typing import Dict, Any, List
import argparse
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, '/app/backend')

class RAGIntegrationTester:
    """Test class for comprehensive RAG integration testing"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api"
        self.test_results = []
        
    async def test_health_check(self) -> Dict[str, Any]:
        """Test basic health endpoint"""
        print("🏥 Testing health check...")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.api_base}/health")
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "test": "health_check",
                        "status": "PASS",
                        "message": f"Health check successful: {data.get('status')}",
                        "data": data
                    }
                else:
                    return {
                        "test": "health_check",
                        "status": "FAIL",
                        "message": f"Health check failed with status {response.status_code}"
                    }
        except Exception as e:
            return {
                "test": "health_check",
                "status": "FAIL",
                "message": f"Health check error: {str(e)}"
            }
    
    async def test_ollama_integration(self) -> Dict[str, Any]:
        """Test OLLAMA connection and model availability"""
        print("🤖 Testing OLLAMA integration...")
        try:
            # Test direct OLLAMA connection
            async with httpx.AsyncClient(timeout=30.0) as client:
                ollama_response = await client.get("http://localhost:11434/api/tags")
                
                if ollama_response.status_code == 200:
                    models = ollama_response.json().get("models", [])
                    model_names = [model["name"] for model in models]
                    
                    # Check if CodeLlama is available
                    has_codellama = any("codellama" in name.lower() for name in model_names)
                    
                    return {
                        "test": "ollama_integration",
                        "status": "PASS" if has_codellama else "WARN",
                        "message": f"OLLAMA connected. Models: {model_names}. CodeLlama available: {has_codellama}",
                        "data": {
                            "models": model_names,
                            "has_codellama": has_codellama
                        }
                    }
                else:
                    return {
                        "test": "ollama_integration",
                        "status": "FAIL",
                        "message": "OLLAMA not accessible"
                    }
                    
        except Exception as e:
            return {
                "test": "ollama_integration",
                "status": "FAIL",
                "message": f"OLLAMA connection error: {str(e)}"
            }
    
    async def test_gitlab_connection(self, gitlab_config: Dict[str, str]) -> Dict[str, Any]:
        """Test GitLab API connection with real credentials"""
        print("🦊 Testing GitLab integration...")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.api_base}/test-connections",
                    json=gitlab_config
                )
                
                if response.status_code == 200:
                    data = response.json()
                    gitlab_status = data.get("gitlab", {})
                    
                    if gitlab_status.get("status") == "connected":
                        return {
                            "test": "gitlab_connection",
                            "status": "PASS",
                            "message": f"GitLab connected as user: {gitlab_status.get('user')}",
                            "data": gitlab_status
                        }
                    else:
                        return {
                            "test": "gitlab_connection",
                            "status": "FAIL",
                            "message": f"GitLab connection failed: {gitlab_status.get('message')}"
                        }
                else:
                    return {
                        "test": "gitlab_connection",
                        "status": "FAIL",
                        "message": f"Connection test failed with status {response.status_code}"
                    }
                    
        except Exception as e:
            return {
                "test": "gitlab_connection",
                "status": "FAIL",
                "message": f"GitLab test error: {str(e)}"
            }
    
    async def test_repository_processing(self, gitlab_config: Dict[str, str]) -> Dict[str, Any]:
        """Test full repository processing workflow"""
        print("📁 Testing repository processing...")
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                # Start repository processing
                response = await client.post(
                    f"{self.api_base}/process-repository",
                    json={"config": gitlab_config}
                )
                
                if response.status_code != 200:
                    return {
                        "test": "repository_processing",
                        "status": "FAIL",
                        "message": f"Failed to start processing: {response.status_code}"
                    }
                
                # Poll for completion
                max_polls = 60  # 5 minutes max
                poll_count = 0
                
                while poll_count < max_polls:
                    status_response = await client.get(f"{self.api_base}/status")
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        current_status = status_data.get("status")
                        
                        print(f"   Status: {current_status} - {status_data.get('message', '')}")
                        
                        if current_status == "completed":
                            # Get repository info
                            info_response = await client.get(f"{self.api_base}/repository-info")
                            repo_info = info_response.json() if info_response.status_code == 200 else {}
                            
                            return {
                                "test": "repository_processing",
                                "status": "PASS",
                                "message": f"Repository processed successfully. {repo_info.get('total_chunks', 0)} chunks created",
                                "data": {
                                    "processing_status": status_data,
                                    "repository_info": repo_info
                                }
                            }
                        
                        elif current_status == "error":
                            return {
                                "test": "repository_processing",
                                "status": "FAIL",
                                "message": f"Processing failed: {status_data.get('message')}"
                            }
                    
                    poll_count += 1
                    await asyncio.sleep(5)  # Wait 5 seconds between polls
                
                return {
                    "test": "repository_processing",
                    "status": "FAIL",
                    "message": "Processing timed out after 5 minutes"
                }
                
        except Exception as e:
            return {
                "test": "repository_processing",
                "status": "FAIL",
                "message": f"Repository processing error: {str(e)}"
            }
    
    async def test_code_suggestions(self) -> Dict[str, Any]:
        """Test code suggestion generation"""
        print("💡 Testing code suggestions...")
        try:
            # Test different suggestion types
            test_queries = [
                {
                    "query": "How to handle errors in Ansible tasks?",
                    "suggestion_type": "general"
                },
                {
                    "query": "This task is failing with permission denied",
                    "suggestion_type": "bugfix"
                },
                {
                    "query": "Add logging functionality to this Python module",
                    "suggestion_type": "feature"
                },
                {
                    "query": "Check for security vulnerabilities in SSH configuration",
                    "suggestion_type": "security"
                }
            ]
            
            results = []
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                for test_query in test_queries:
                    response = await client.post(
                        f"{self.api_base}/suggest",
                        json=test_query
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        suggestion_length = len(data.get("suggestion", ""))
                        context_count = len(data.get("context_chunks", []))
                        
                        results.append({
                            "query_type": test_query["suggestion_type"],
                            "suggestion_length": suggestion_length,
                            "context_chunks": context_count,
                            "status": "SUCCESS"
                        })
                    else:
                        results.append({
                            "query_type": test_query["suggestion_type"],
                            "status": "FAILED",
                            "error": response.text
                        })
            
            successful_tests = len([r for r in results if r["status"] == "SUCCESS"])
            
            return {
                "test": "code_suggestions",
                "status": "PASS" if successful_tests == len(test_queries) else "PARTIAL",
                "message": f"Code suggestions: {successful_tests}/{len(test_queries)} successful",
                "data": {"results": results}
            }
            
        except Exception as e:
            return {
                "test": "code_suggestions",
                "status": "FAIL",
                "message": f"Code suggestions error: {str(e)}"
            }
    
    async def test_analytics_features(self) -> Dict[str, Any]:
        """Test analytics and advanced features"""
        print("📊 Testing analytics features...")
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Test code quality analysis
                quality_response = await client.post(f"{self.api_base}/analyze-code-quality")
                quality_success = quality_response.status_code == 200
                
                # Test similar code search
                similar_response = await client.post(
                    f"{self.api_base}/search-similar-code",
                    json={"code_snippet": "def main():\n    pass"}
                )
                similar_success = similar_response.status_code == 200
                
                successful_features = sum([quality_success, similar_success])
                
                return {
                    "test": "analytics_features",
                    "status": "PASS" if successful_features == 2 else "PARTIAL",
                    "message": f"Analytics features: {successful_features}/2 working",
                    "data": {
                        "quality_analysis": quality_success,
                        "similar_code_search": similar_success
                    }
                }
                
        except Exception as e:
            return {
                "test": "analytics_features",
                "status": "FAIL",
                "message": f"Analytics features error: {str(e)}"
            }
    
    async def run_full_test_suite(self, gitlab_config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Run the complete test suite"""
        print("🚀 Starting comprehensive RAG integration tests...\n")
        
        # Test sequence
        tests = [
            self.test_health_check(),
            self.test_ollama_integration(),
            self.test_gitlab_connection(gitlab_config),
            self.test_repository_processing(gitlab_config),
            self.test_code_suggestions(),
            self.test_analytics_features()
        ]
        
        results = []
        for test in tests:
            result = await test
            results.append(result)
            
            # Print result
            status_emoji = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "PARTIAL": "🔶"}
            emoji = status_emoji.get(result["status"], "❓")
            print(f"{emoji} {result['test']}: {result['message']}")
            
            # Stop on critical failures
            if result["test"] in ["health_check", "gitlab_connection"] and result["status"] == "FAIL":
                print(f"\n💥 Critical test failed: {result['test']}. Stopping test suite.")
                break
        
        return results
    
    def generate_test_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive test report"""
        total_tests = len(results)
        passed_tests = len([r for r in results if r["status"] == "PASS"])
        failed_tests = len([r for r in results if r["status"] == "FAIL"])
        warning_tests = len([r for r in results if r["status"] in ["WARN", "PARTIAL"]])
        
        report = f"""
# RAG Code Assistant Integration Test Report

## Summary
- **Total Tests**: {total_tests}
- **Passed**: {passed_tests} ✅
- **Failed**: {failed_tests} ❌
- **Warnings**: {warning_tests} ⚠️
- **Success Rate**: {(passed_tests/total_tests)*100:.1f}%

## Test Results

"""
        
        for result in results:
            status_emoji = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "PARTIAL": "🔶"}
            emoji = status_emoji.get(result["status"], "❓")
            
            report += f"### {emoji} {result['test'].replace('_', ' ').title()}\n"
            report += f"**Status**: {result['status']}\n"
            report += f"**Message**: {result['message']}\n"
            
            if "data" in result:
                report += f"**Details**: {json.dumps(result['data'], indent=2)}\n"
            
            report += "\n"
        
        # Add recommendations
        report += "## Recommendations\n\n"
        
        if failed_tests > 0:
            report += "❌ **Critical Issues Found:**\n"
            for result in results:
                if result["status"] == "FAIL":
                    report += f"- Fix {result['test']}: {result['message']}\n"
            report += "\n"
        
        if warning_tests > 0:
            report += "⚠️ **Warnings:**\n"
            for result in results:
                if result["status"] in ["WARN", "PARTIAL"]:
                    report += f"- {result['test']}: {result['message']}\n"
            report += "\n"
        
        if passed_tests == total_tests:
            report += "🎉 **All tests passed! Your RAG Code Assistant is ready for production use.**\n"
        
        return report

async def main():
    parser = argparse.ArgumentParser(description='RAG Code Assistant Integration Testing')
    parser.add_argument('--gitlab-url', default='https://gitlab.com', help='GitLab URL')
    parser.add_argument('--gitlab-token', required=True, help='GitLab API token')
    parser.add_argument('--repository', required=True, help='Repository path (user/repo)')
    parser.add_argument('--branch', default='main', help='Repository branch')
    parser.add_argument('--backend-url', default='http://localhost:8001', help='Backend URL')
    parser.add_argument('--output', help='Output file for test report')
    
    args = parser.parse_args()
    
    # GitLab configuration
    gitlab_config = {
        "gitlab_url": args.gitlab_url,
        "api_token": args.gitlab_token,
        "repository_path": args.repository,
        "branch": args.branch
    }
    
    # Run tests
    tester = RAGIntegrationTester(args.backend_url)
    results = await tester.run_full_test_suite(gitlab_config)
    
    # Generate report
    report = tester.generate_test_report(results)
    print("\n" + "="*60)
    print(report)
    
    # Save report if output specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"📄 Test report saved to: {args.output}")
    
    # Exit with appropriate code
    failed_count = len([r for r in results if r["status"] == "FAIL"])
    sys.exit(0 if failed_count == 0 else 1)

if __name__ == "__main__":
    asyncio.run(main())
