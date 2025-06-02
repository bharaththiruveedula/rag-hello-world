import requests
import unittest
import sys
import time
from datetime import datetime

class RAGCodeAssistantTester:
    def __init__(self, base_url="https://02ddbe4f-34dc-4c31-b103-f4cd54ad3e5b.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def run_test(self, name, method, endpoint, expected_status, data=None):
        """Run a single API test"""
        url = f"{self.base_url}/api/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                result = {"name": name, "status": "passed", "response": response.json() if response.text else {}}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                result = {"name": name, "status": "failed", "expected": expected_status, "got": response.status_code}

            self.test_results.append(result)
            return success, response.json() if response.text and success else {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.test_results.append({"name": name, "status": "error", "message": str(e)})
            return False, {}

    def test_health_endpoint(self):
        """Test the health endpoint"""
        return self.run_test(
            "Health Endpoint",
            "GET",
            "health",
            200
        )

    def test_status_endpoint(self):
        """Test the status endpoint"""
        return self.run_test(
            "Status Endpoint",
            "GET",
            "status",
            200
        )

    def test_connections(self):
        """Test the connections endpoint with mock GitLab config"""
        mock_config = {
            "gitlab_url": "https://gitlab.com",
            "api_token": "mock_token",
            "repository_path": "mock/repo",
            "branch": "main"
        }
        
        return self.run_test(
            "Test Connections",
            "POST",
            "test-connections",
            200,
            data=mock_config
        )

    def test_repository_info(self):
        """Test the repository info endpoint"""
        return self.run_test(
            "Repository Info",
            "GET",
            "repository-info",
            200
        )
    
    def test_code_suggestion_general(self):
        """Test code suggestion with general type"""
        return self.run_test(
            "Code Suggestion - General",
            "POST",
            "suggest",
            200,
            data={
                "query": "How to handle errors in Ansible tasks?",
                "suggestion_type": "general"
            }
        )
    
    def test_code_suggestion_bugfix(self):
        """Test code suggestion with bugfix type"""
        return self.run_test(
            "Code Suggestion - Bugfix",
            "POST",
            "suggest",
            200,
            data={
                "query": "Fix permission denied error in file access",
                "suggestion_type": "bugfix"
            }
        )
    
    def test_code_suggestion_security(self):
        """Test code suggestion with security type"""
        return self.run_test(
            "Code Suggestion - Security",
            "POST",
            "suggest",
            200,
            data={
                "query": "Check for security vulnerabilities in SSH configuration",
                "suggestion_type": "security"
            }
        )
    
    def test_code_suggestion_performance(self):
        """Test code suggestion with performance type"""
        return self.run_test(
            "Code Suggestion - Performance",
            "POST",
            "suggest",
            200,
            data={
                "query": "Optimize database query performance",
                "suggestion_type": "performance"
            }
        )
    
    def test_code_suggestion_documentation(self):
        """Test code suggestion with documentation type"""
        return self.run_test(
            "Code Suggestion - Documentation",
            "POST",
            "suggest",
            200,
            data={
                "query": "Generate documentation for this Python function",
                "suggestion_type": "documentation"
            }
        )
    
    def test_code_suggestion_refactor(self):
        """Test code suggestion with refactor type"""
        return self.run_test(
            "Code Suggestion - Refactor",
            "POST",
            "suggest",
            200,
            data={
                "query": "Refactor this code to improve maintainability",
                "suggestion_type": "refactor"
            }
        )
    
    def test_analyze_code_quality(self):
        """Test code quality analysis endpoint"""
        return self.run_test(
            "Code Quality Analysis",
            "POST",
            "analyze-code-quality",
            200
        )
    
    def test_search_similar_code(self):
        """Test similar code search endpoint"""
        return self.run_test(
            "Similar Code Search",
            "POST",
            "search-similar-code",
            200,
            data={
                "code_snippet": "def process_data(data):\n    result = []\n    for item in data:\n        result.append(item * 2)\n    return result"
            }
        )
    
    def test_jira_suggest(self):
        """Test JIRA-based code suggestion endpoint"""
        return self.run_test(
            "JIRA Suggestion",
            "POST",
            "jira-suggest",
            200,
            data={
                "ticket_id": "PROJ-123",
                "suggestion_type": "general"
            }
        )
    
    def test_jira_suggest_bugfix(self):
        """Test JIRA-based code suggestion with bugfix type"""
        return self.run_test(
            "JIRA Suggestion - Bugfix",
            "POST",
            "jira-suggest",
            200,
            data={
                "ticket_id": "PROJ-123",
                "suggestion_type": "bugfix"
            }
        )
    
    def test_jira_suggest_feature(self):
        """Test JIRA-based code suggestion with feature type"""
        return self.run_test(
            "JIRA Suggestion - Feature",
            "POST",
            "jira-suggest",
            200,
            data={
                "ticket_id": "PROJ-123",
                "suggestion_type": "feature"
            }
        )
    
    def test_jira_suggest_invalid_ticket(self):
        """Test JIRA-based code suggestion with invalid ticket ID"""
        return self.run_test(
            "JIRA Suggestion - Invalid Ticket",
            "POST",
            "jira-suggest",
            404,  # Should return 404 for non-existent ticket
            data={
                "ticket_id": "INVALID-999",
                "suggestion_type": "general"
            }
        )

    def test_error_handling(self):
        """Test error handling for invalid requests"""
        # Test missing code snippet
        success, _ = self.run_test(
            "Error Handling - Missing Code Snippet",
            "POST",
            "search-similar-code",
            400,
            data={}
        )
        
        # Test invalid suggestion type
        success2, _ = self.run_test(
            "Error Handling - Invalid Suggestion Type",
            "POST",
            "suggest",
            200,  # Should still work with default type
            data={
                "query": "Help with code",
                "suggestion_type": "invalid_type"
            }
        )
        
        return success and success2
        
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*50)
        print(f"📊 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        print("="*50)
        
        for result in self.test_results:
            status_icon = "✅" if result["status"] == "passed" else "❌"
            print(f"{status_icon} {result['name']}")
            
            if result["status"] == "passed" and "response" in result:
                # Print a truncated version of the response for readability
                response_str = str(result["response"])
                if len(response_str) > 200:
                    response_str = response_str[:200] + "..."
                print(f"   Response: {response_str}")
            elif result["status"] == "failed":
                print(f"   Expected status: {result['expected']}, Got: {result['got']}")
            elif result["status"] == "error":
                print(f"   Error: {result['message']}")
                
        print("="*50)
        return self.tests_passed == self.tests_run

def main():
    # Setup
    tester = RAGCodeAssistantTester()
    
    # Run basic API tests
    tester.test_health_endpoint()
    tester.test_status_endpoint()
    tester.test_connections()
    tester.test_repository_info()
    
    # Test new suggestion types
    tester.test_code_suggestion_general()
    tester.test_code_suggestion_bugfix()
    tester.test_code_suggestion_security()
    tester.test_code_suggestion_performance()
    tester.test_code_suggestion_documentation()
    tester.test_code_suggestion_refactor()
    
    # Test JIRA integration
    tester.test_jira_suggest()
    tester.test_jira_suggest_bugfix()
    tester.test_jira_suggest_feature()
    tester.test_jira_suggest_invalid_ticket()
    
    # Test analytics features
    tester.test_analyze_code_quality()
    tester.test_search_similar_code()
    
    # Test error handling
    tester.test_error_handling()
    
    # Print results
    success = tester.print_summary()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
