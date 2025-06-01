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

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*50)
        print(f"📊 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        print("="*50)
        
        for result in self.test_results:
            status_icon = "✅" if result["status"] == "passed" else "❌"
            print(f"{status_icon} {result['name']}")
            
            if result["status"] == "passed" and "response" in result:
                print(f"   Response: {result['response']}")
            elif result["status"] == "failed":
                print(f"   Expected status: {result['expected']}, Got: {result['got']}")
            elif result["status"] == "error":
                print(f"   Error: {result['message']}")
                
        print("="*50)
        return self.tests_passed == self.tests_run

def main():
    # Setup
    tester = RAGCodeAssistantTester()
    
    # Run tests
    tester.test_health_endpoint()
    tester.test_status_endpoint()
    tester.test_connections()
    tester.test_repository_info()
    
    # Print results
    success = tester.print_summary()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
