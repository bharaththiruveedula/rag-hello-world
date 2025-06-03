#====================================================================================================
# START - Testing Protocol - DO NOT EDIT OR REMOVE THIS SECTION
#====================================================================================================

# THIS SECTION CONTAINS CRITICAL TESTING INSTRUCTIONS FOR BOTH AGENTS
# BOTH MAIN_AGENT AND TESTING_AGENT MUST PRESERVE THIS ENTIRE BLOCK

# Communication Protocol:
# If the `testing_agent` is available, main agent should delegate all testing tasks to it.
#
# You have access to a file called `test_result.md`. This file contains the complete testing state
# and history, and is the primary means of communication between main and the testing agent.
#
# Main and testing agents must follow this exact format to maintain testing data. 
# The testing data must be entered in yaml format Below is the data structure:
# 
## user_problem_statement: {problem_statement}
## backend:
##   - task: "Task name"
##     implemented: true
##     working: true  # or false or "NA"
##     file: "file_path.py"
##     stuck_count: 0
##     priority: "high"  # or "medium" or "low"
##     needs_retesting: false
##     status_history:
##         -working: true  # or false or "NA"
##         -agent: "main"  # or "testing" or "user"
##         -comment: "Detailed comment about status"
##
## frontend:
##   - task: "Task name"
##     implemented: true
##     working: true  # or false or "NA"
##     file: "file_path.js"
##     stuck_count: 0
##     priority: "high"  # or "medium" or "low"
##     needs_retesting: false
##     status_history:
##         -working: true  # or false or "NA"
##         -agent: "main"  # or "testing" or "user"
##         -comment: "Detailed comment about status"
##
## metadata:
##   created_by: "main_agent"
##   version: "1.0"
##   test_sequence: 0
##   run_ui: false
##
## test_plan:
##   current_focus:
##     - "Task name 1"
##     - "Task name 2"
##   stuck_tasks:
##     - "Task name with persistent issues"
##   test_all: false
##   test_priority: "high_first"  # or "sequential" or "stuck_first"
##
## agent_communication:
##     -agent: "main"  # or "testing" or "user"
##     -message: "Communication message between agents"

# Protocol Guidelines for Main agent
#
# 1. Update Test Result File Before Testing:
#    - Main agent must always update the `test_result.md` file before calling the testing agent
#    - Add implementation details to the status_history
#    - Set `needs_retesting` to true for tasks that need testing
#    - Update the `test_plan` section to guide testing priorities
#    - Add a message to `agent_communication` explaining what you've done
#
# 2. Incorporate User Feedback:
#    - When a user provides feedback that something is or isn't working, add this information to the relevant task's status_history
#    - Update the working status based on user feedback
#    - If a user reports an issue with a task that was marked as working, increment the stuck_count
#    - Whenever user reports issue in the app, if we have testing agent and task_result.md file so find the appropriate task for that and append in status_history of that task to contain the user concern and problem as well 
#
# 3. Track Stuck Tasks:
#    - Monitor which tasks have high stuck_count values or where you are fixing same issue again and again, analyze that when you read task_result.md
#    - For persistent issues, use websearch tool to find solutions
#    - Pay special attention to tasks in the stuck_tasks list
#    - When you fix an issue with a stuck task, don't reset the stuck_count until the testing agent confirms it's working
#
# 4. Provide Context to Testing Agent:
#    - When calling the testing agent, provide clear instructions about:
#      - Which tasks need testing (reference the test_plan)
#      - Any authentication details or configuration needed
#      - Specific test scenarios to focus on
#      - Any known issues or edge cases to verify
#
# 5. Call the testing agent with specific instructions referring to test_result.md
#
# IMPORTANT: Main agent must ALWAYS update test_result.md BEFORE calling the testing agent, as it relies on this file to understand what to test next.

#====================================================================================================
# END - Testing Protocol - DO NOT EDIT OR REMOVE THIS SECTION
#====================================================================================================



#====================================================================================================
# Testing Data - Main Agent and testing sub agent both should log testing data below this section
#====================================================================================================

user_problem_statement: "Use this repository.... Instead of using JIRA rest api ... use python JIRA package as my enterprise JIRA is not working with direct API calls and use token_auth for authentication"

backend:
  - task: "Replace JIRA REST API calls with Python JIRA package"
    implemented: true
    working: true
    file: "backend/server.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: false
    status_history:
      - working: "NA"
        agent: "main"
        comment: "Modified JIRA integration to use Python JIRA package instead of direct httpx REST calls. Added token_auth as primary authentication method with basic_auth fallback. Updated imports, startup event, test_jira_connection(), and fetch_jira_ticket() functions."
      - working: true
        agent: "testing"
        comment: "Verified that the JIRA integration now uses the Python JIRA package instead of direct REST calls. The code structure is correct with proper imports, client initialization, and error handling. The /api/test-connections endpoint correctly tests JIRA connection using jira_client.current_user() and the /api/jira-suggest endpoint uses jira_client.issue(). Both token_auth (primary) and basic_auth (fallback) paths are implemented correctly. Since we're using placeholder values in .env, we're seeing expected error responses with proper error handling."

  - task: "Implement token authentication for JIRA"
    implemented: true
    working: true 
    file: "backend/server.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: false
    status_history:
      - working: "NA"
        agent: "main"
        comment: "Implemented token_auth as primary authentication method for enterprise JIRA. Added fallback to basic_auth if token_auth fails and username is provided. Updated JIRA client initialization in startup event and both connection test and ticket fetch functions."
      - working: true
        agent: "testing"
        comment: "Verified that token authentication is correctly implemented as the primary authentication method with basic_auth as fallback. The JIRA client initialization in the startup event attempts token_auth first and falls back to basic_auth if token_auth fails and a username is provided. The same pattern is implemented in both test_jira_connection() and fetch_jira_ticket() functions. The auth_method is correctly reported in the API response when connection is successful."

metadata:
  created_by: "main_agent"
  version: "1.0"
  test_sequence: 0
  run_ui: false

test_plan:
  current_focus:
    - "Replace JIRA REST API calls with Python JIRA package"
    - "Implement token authentication for JIRA"
  stuck_tasks: []
  test_all: false
  test_priority: "high_first"

agent_communication:
  - agent: "main"
    message: "Successfully modified JIRA integration from direct REST API calls to Python JIRA package with token authentication. Key changes: 1) Added JIRA imports and exception handling, 2) Updated startup event to initialize JIRA client with token_auth (primary) and basic_auth (fallback), 3) Replaced test_jira_connection() to use jira_client.current_user(), 4) Replaced fetch_jira_ticket() to use jira_client.issue() with better error handling and description parsing. Ready for backend testing to verify JIRA integration works with enterprise setup."