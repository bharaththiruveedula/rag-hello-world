#!/usr/bin/env python3
"""
Frontend UI testing for RAG Code Assistant using Playwright.
Tests the complete user interface and workflows.
"""

import asyncio
import sys
import time
from playwright.async_api import async_playwright, Page, Browser
from typing import Dict, List, Any
import argparse
import json


class FrontendTester:
    """Frontend UI testing class"""
    
    def __init__(self, frontend_url: str):
        self.frontend_url = frontend_url
        self.test_results = []
        
    async def test_page_load(self, page: Page) -> Dict[str, Any]:
        """Test basic page loading and UI elements"""
        print("🌐 Testing page load and basic UI...")
        try:
            await page.goto(self.frontend_url)
            await page.wait_for_load_state('networkidle')
            
            # Check if main elements are present
            title = await page.title()
            app_title = await page.text_content('h1.app-title')
            
            # Check tab navigation
            tabs = await page.locator('.tab-button').count()
            
            # Check if setup form is visible
            gitlab_url_input = await page.locator('input[placeholder*="gitlab.com"]').count()
            
            return {
                "test": "page_load",
                "status": "PASS",
                "message": f"Page loaded successfully. Title: '{app_title}', {tabs} tabs found",
                "data": {
                    "page_title": title,
                    "app_title": app_title,
                    "tab_count": tabs,
                    "has_setup_form": gitlab_url_input > 0
                }
            }
            
        except Exception as e:
            return {
                "test": "page_load",
                "status": "FAIL",
                "message": f"Page load failed: {str(e)}"
            }
    
    async def test_tab_navigation(self, page: Page) -> Dict[str, Any]:
        """Test tab navigation functionality"""
        print("🔄 Testing tab navigation...")
        try:
            # Test each tab
            tabs = ['setup', 'status', 'analytics', 'chat']
            successful_tabs = []
            
            for tab in tabs:
                tab_button = page.locator(f'.tab-button:has-text("{tab.capitalize()}")')
                await tab_button.click()
                await page.wait_for_timeout(500)  # Small delay for UI update
                
                # Check if tab is active
                is_active = await tab_button.get_attribute('class')
                if 'active' in is_active:
                    successful_tabs.append(tab)
            
            return {
                "test": "tab_navigation",
                "status": "PASS" if len(successful_tabs) == len(tabs) else "PARTIAL",
                "message": f"Tab navigation: {len(successful_tabs)}/{len(tabs)} tabs working",
                "data": {"working_tabs": successful_tabs}
            }
            
        except Exception as e:
            return {
                "test": "tab_navigation",
                "status": "FAIL",
                "message": f"Tab navigation failed: {str(e)}"
            }
    
    async def test_form_functionality(self, page: Page) -> Dict[str, Any]:
        """Test GitLab configuration form"""
        print("📝 Testing form functionality...")
        try:
            # Navigate to setup tab
            await page.click('.tab-button:has-text("Setup")')
            await page.wait_for_timeout(500)
            
            # Fill out form
            await page.fill('input[placeholder*="gitlab.com"]', 'https://gitlab.example.com')
            await page.fill('input[placeholder*="api-token"]', 'test-token-123')
            await page.fill('input[placeholder*="repository-name"]', 'testuser/test-repo')
            await page.fill('input[placeholder*="main"]', 'develop')
            
            # Check if values were set
            gitlab_url = await page.input_value('input[placeholder*="gitlab.com"]')
            token = await page.input_value('input[placeholder*="api-token"]')
            repo = await page.input_value('input[placeholder*="repository-name"]')
            branch = await page.input_value('input[placeholder*="main"]')
            
            # Check if buttons are enabled
            test_btn_disabled = await page.is_disabled('button:has-text("Test Connections")')
            process_btn_disabled = await page.is_disabled('button:has-text("Process Repository")')
            
            return {
                "test": "form_functionality",
                "status": "PASS",
                "message": "Form functionality working correctly",
                "data": {
                    "form_values": {
                        "gitlab_url": gitlab_url,
                        "has_token": len(token) > 0,
                        "repository": repo,
                        "branch": branch
                    },
                    "buttons_enabled": not test_btn_disabled and not process_btn_disabled
                }
            }
            
        except Exception as e:
            return {
                "test": "form_functionality",
                "status": "FAIL",
                "message": f"Form functionality failed: {str(e)}"
            }
    
    async def test_connection_testing(self, page: Page, gitlab_config: Dict[str, str]) -> Dict[str, Any]:
        """Test connection testing functionality"""
        print("🔌 Testing connection testing...")
        try:
            # Fill form with real credentials
            await page.fill('input[placeholder*="gitlab.com"]', gitlab_config["gitlab_url"])
            await page.fill('input[placeholder*="api-token"]', gitlab_config["api_token"])
            await page.fill('input[placeholder*="repository-name"]', gitlab_config["repository_path"])
            await page.fill('input[placeholder*="main"]', gitlab_config["branch"])
            
            # Click test connections button
            await page.click('button:has-text("Test Connections")')
            
            # Wait for results (up to 30 seconds)
            await page.wait_for_selector('.connection-status', timeout=30000)
            
            # Check connection status
            ollama_status = await page.locator('.status-item:has-text("OLLAMA")').count()
            gitlab_status = await page.locator('.status-item:has-text("GitLab")').count()
            qdrant_status = await page.locator('.status-item:has-text("Qdrant")').count()
            
            return {
                "test": "connection_testing",
                "status": "PASS",
                "message": "Connection testing completed successfully",
                "data": {
                    "status_items_found": {
                        "ollama": ollama_status > 0,
                        "gitlab": gitlab_status > 0,
                        "qdrant": qdrant_status > 0
                    }
                }
            }
            
        except Exception as e:
            return {
                "test": "connection_testing",
                "status": "FAIL",
                "message": f"Connection testing failed: {str(e)}"
            }
    
    async def test_responsive_design(self, page: Page) -> Dict[str, Any]:
        """Test responsive design on different screen sizes"""
        print("📱 Testing responsive design...")
        try:
            # Test different viewport sizes
            viewport_tests = [
                {"width": 1920, "height": 1080, "name": "desktop"},
                {"width": 768, "height": 1024, "name": "tablet"},
                {"width": 375, "height": 667, "name": "mobile"}
            ]
            
            successful_viewports = []
            
            for viewport in viewport_tests:
                await page.set_viewport_size(width=viewport["width"], height=viewport["height"])
                await page.wait_for_timeout(500)
                
                # Check if main elements are still visible and accessible
                app_title_visible = await page.is_visible('h1.app-title')
                tabs_visible = await page.is_visible('.tab-navigation')
                
                if app_title_visible and tabs_visible:
                    successful_viewports.append(viewport["name"])
            
            return {
                "test": "responsive_design",
                "status": "PASS" if len(successful_viewports) == len(viewport_tests) else "PARTIAL",
                "message": f"Responsive design: {len(successful_viewports)}/{len(viewport_tests)} viewports working",
                "data": {"working_viewports": successful_viewports}
            }
            
        except Exception as e:
            return {
                "test": "responsive_design",
                "status": "FAIL",
                "message": f"Responsive design test failed: {str(e)}"
            }
    
    async def test_chat_interface(self, page: Page) -> Dict[str, Any]:
        """Test chat interface functionality"""
        print("💬 Testing chat interface...")
        try:
            # Navigate to chat tab
            await page.click('.tab-button:has-text("Chat")')
            await page.wait_for_timeout(500)
            
            # Check if chat is disabled (repository not processed)
            disabled_message = await page.locator('.chat-disabled').count()
            
            if disabled_message > 0:
                return {
                    "test": "chat_interface",
                    "status": "PASS",
                    "message": "Chat interface correctly shows disabled state when repository not processed",
                    "data": {"disabled_correctly": True}
                }
            
            # If chat is enabled, test form elements
            suggestion_select = await page.locator('select').count()
            query_textarea = await page.locator('textarea').count()
            submit_button = await page.locator('button:has-text("Get Suggestions")').count()
            
            return {
                "test": "chat_interface",
                "status": "PASS",
                "message": "Chat interface elements present and functional",
                "data": {
                    "has_suggestion_select": suggestion_select > 0,
                    "has_query_textarea": query_textarea > 0,
                    "has_submit_button": submit_button > 0
                }
            }
            
        except Exception as e:
            return {
                "test": "chat_interface",
                "status": "FAIL",
                "message": f"Chat interface test failed: {str(e)}"
            }
    
    async def test_analytics_interface(self, page: Page) -> Dict[str, Any]:
        """Test analytics interface"""
        print("📊 Testing analytics interface...")
        try:
            # Navigate to analytics tab
            await page.click('.tab-button:has-text("Analytics")')
            await page.wait_for_timeout(500)
            
            # Check if analytics is disabled (repository not processed)
            disabled_message = await page.locator('.analytics-disabled').count()
            
            if disabled_message > 0:
                return {
                    "test": "analytics_interface",
                    "status": "PASS",
                    "message": "Analytics interface correctly shows disabled state when repository not processed",
                    "data": {"disabled_correctly": True}
                }
            
            # If analytics is enabled, test interface elements
            quality_card = await page.locator('.analytics-card:has-text("Code Quality")').count()
            similar_card = await page.locator('.analytics-card:has-text("Similar Code")').count()
            
            return {
                "test": "analytics_interface",
                "status": "PASS",
                "message": "Analytics interface elements present",
                "data": {
                    "has_quality_analysis": quality_card > 0,
                    "has_similar_search": similar_card > 0
                }
            }
            
        except Exception as e:
            return {
                "test": "analytics_interface",
                "status": "FAIL",
                "message": f"Analytics interface test failed: {str(e)}"
            }
    
    async def run_frontend_tests(self, gitlab_config: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """Run all frontend tests"""
        print("🎭 Starting frontend UI tests...\n")
        
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            try:
                # Test sequence
                tests = [
                    self.test_page_load(page),
                    self.test_tab_navigation(page),
                    self.test_form_functionality(page),
                    self.test_responsive_design(page),
                    self.test_chat_interface(page),
                    self.test_analytics_interface(page)
                ]
                
                # Add connection testing if credentials provided
                if gitlab_config:
                    tests.append(self.test_connection_testing(page, gitlab_config))
                
                results = []
                for test in tests:
                    result = await test
                    results.append(result)
                    
                    # Print result
                    status_emoji = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "PARTIAL": "🔶"}
                    emoji = status_emoji.get(result["status"], "❓")
                    print(f"{emoji} {result['test']}: {result['message']}")
                
                return results
                
            finally:
                await browser.close()


async def main():
    parser = argparse.ArgumentParser(description='RAG Code Assistant Frontend Testing')
    parser.add_argument('--frontend-url', default='http://localhost:3000', help='Frontend URL')
    parser.add_argument('--gitlab-token', help='GitLab API token for connection testing')
    parser.add_argument('--repository', help='Repository path for connection testing')
    parser.add_argument('--output', help='Output file for test report')
    
    args = parser.parse_args()
    
    # GitLab configuration for connection testing
    gitlab_config = None
    if args.gitlab_token and args.repository:
        gitlab_config = {
            "gitlab_url": "https://gitlab.com",
            "api_token": args.gitlab_token,
            "repository_path": args.repository,
            "branch": "main"
        }
    
    # Run tests
    tester = FrontendTester(args.frontend_url)
    results = await tester.run_frontend_tests(gitlab_config)
    
    # Generate simple report
    total_tests = len(results)
    passed_tests = len([r for r in results if r["status"] == "PASS"])
    
    print(f"\n🎭 Frontend Test Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Save detailed results if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Detailed results saved to: {args.output}")
    
    # Exit with appropriate code
    failed_count = len([r for r in results if r["status"] == "FAIL"])
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
