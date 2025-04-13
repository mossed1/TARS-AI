import asyncio
import json
import os
import shutil
from typing import Dict, List, Optional, Any
from contextlib import suppress

import requests
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class Configuration:
    """Manages configuration and environment variables for the MCP client."""
    def __init__(self) -> None:
        self.load_env()
        self.api_key = os.getenv("GROQ_API_KEY")
        # self.api_key = os.getenv("GITHUB_API_KEY")

    @staticmethod
    def load_env() -> None:
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        # For this refactoring, we simply return a fixed value.
        return "xyz"


class Server:
    """Manages MCP server connections and tool execution."""
    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.stdio_context: Optional[Any] = None
        self.session: Optional[ClientSession] = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.capabilities: Optional[Dict[str, Any]] = None

    async def initialize(self) -> None:
        server_params = StdioServerParameters(
            command=shutil.which("npx") if self.config['command'] == "npx" else self.config['command'],
            args=self.config['args'],
            env={**os.environ, **self.config['env']} if self.config.get('env') else None
        )
        try:
            self.stdio_context = stdio_client(server_params)
            read, write = await self.stdio_context.__aenter__()
            self.session = ClientSession(read, write)
            await self.session.__aenter__()
            self.capabilities = await self.session.initialize()
        except Exception as e:
            print(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[Any]:
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        tools_response = await self.session.list_tools()
        tools = []
        supports_progress = self.capabilities and 'progress' in self.capabilities
        if supports_progress:
            print(f"Server {self.name} supports progress tracking")
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == 'tools':
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))
                    if supports_progress:
                        print(f"Tool '{tool.name}' will support progress tracking")
        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        retries: int = 2,
        delay: float = 1.0
    ) -> Any:
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                supports_progress = self.capabilities and 'progress' in self.capabilities
                if supports_progress:
                    print(f"Executing {tool_name} with progress tracking...")
                    result = await self.session.call_tool(
                        tool_name,
                        arguments,
                        progress_token=f"{tool_name}_execution"
                    )
                else:
                    print(f"Executing {tool_name}...")
                    result = await self.session.call_tool(tool_name, arguments)
                return result

            except Exception as e:
                attempt += 1
                print(f"Error executing tool: {e}. Attempt {attempt} of {retries}.")
                if attempt < retries:
                    print(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    print("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        async with self._cleanup_lock:
            # Cleanup session
            if self.session:
                with suppress(asyncio.CancelledError):
                    try:
                        await self.session.__aexit__(None, None, None)
                    except Exception as e:
                        print(f"Warning during session cleanup for {self.name}: {e}")
                self.session = None

            # Cleanup stdio context
            if self.stdio_context:
                with suppress(asyncio.CancelledError):
                    try:
                        await self.stdio_context.__aexit__(None, None, None)
                    except (RuntimeError, asyncio.CancelledError) as e:
                        print(f"Note: Normal shutdown message for {self.name}: {e}")
                    except Exception as e:
                        print(f"Warning during stdio cleanup for {self.name}: {e}")
                self.stdio_context = None


class Tool:
    """Represents a tool with its properties and formatting."""
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        args_desc = []
        if 'properties' in self.input_schema:
            for param_name, param_info in self.input_schema['properties'].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get('required', []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)
        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """Manages communication with the LLM provider."""
    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        url = "http://192.168.2.57:1234/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "messages": messages,
            "model": "llama-3.2-90b-vision-preview",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False,
            "stop": None
        }
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            error_message = f"Error getting LLM response: {str(e)}"
            print(error_message)
            if e.response is not None:
                print(f"Status code: {e.response.status_code}")
                print(f"Response details: {e.response.text}")
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""
    def __init__(self, servers: List[Server], llm_client: LLMClient) -> None:
        self.servers: List[Server] = servers
        self.llm_client: LLMClient = llm_client

    async def cleanup_servers(self) -> None:
        cleanup_tasks = [asyncio.create_task(server.cleanup()) for server in self.servers]
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"Warning during final cleanup: {e}")

    async def process_llm_response(self, llm_response: str) -> str:
        """
        If the LLM response is a tool call, execute it and return its output.
        If the response isn’t a tool call, simply return the original response.
        """
        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                print(f"Executing tool: {tool_call['tool']}")
                print(f"With arguments: {tool_call['arguments']}")
                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(tool_call["tool"], tool_call["arguments"])
                            if isinstance(result, dict) and 'progress' in result and 'total' in result:
                                progress = result['progress']
                                total = result['total']
                                print(f"Progress: {progress}/{total} ({(progress/total)*100:.1f}%)")
                            return f"Tool execution result: {result}"
                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            print(error_msg)
                            return error_msg
                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response


async def process_user_prompt(user_prompt: str) -> Optional[str]:
    """
    Process a single user prompt.
    
    This function sets up the servers, initializes the LLM client, and sends the
    user prompt along with a system message containing tool descriptions.
    
    If the LLM response indicates that a tool should be called, the tool is executed
    and its output is returned. If no tool is needed (i.e. the LLM response isn’t a
    valid tool call), the function returns None.
    """
    # Load configuration
    config = Configuration()
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config-MCP.json')
    server_config = config.load_config(config_path)

    # Initialize servers
    servers = [Server(name, srv_config) for name, srv_config in server_config['mcpServers'].items()]
    for server in servers:
        try:
            await server.initialize()
        except Exception as e:
            print(f"Failed to initialize server {server.name}: {e}")
            await asyncio.gather(*(s.cleanup() for s in servers), return_exceptions=True)
            return None

    # Prepare system message with tool descriptions
    all_tools = []
    for server in servers:
        tools = await server.list_tools()
        all_tools.extend(tools)
    tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
    system_message = f"""You are a helpful assistant with access to these tools: 

{tools_description}
Choose the appropriate tool based on the user's question. If no tool is needed, reply directly.

IMPORTANT: When you need to use a tool, you must ONLY respond with the exact JSON object format below, nothing else:
{{
    "tool": "tool-name",
    "arguments": {{
        "argument-name": "value"
    }}
}}

After receiving a tool's response:
1. Transform the raw data into a natural, conversational response
2. Keep responses concise but informative
3. Focus on the most relevant information
4. Use appropriate context from the user's question
5. Avoid simply repeating the raw data

Please use only the tools that are explicitly defined above."""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
    llm_response = config.llm_api_key and LLMClient(config.llm_api_key).get_response(messages) or ""
    
    chat_session = ChatSession(servers, LLMClient(config.llm_api_key))
    tool_result = await chat_session.process_llm_response(llm_response)

    await chat_session.cleanup_servers()

    if tool_result != llm_response:
        return tool_result
    return None


async def main() -> None:
    prompt = input("Enter your prompt: ").strip()
    result = await process_user_prompt(prompt)
    if result is not None:
        print("\nTool call output:")
        print(result)
    else:
        print("\nNo tool call was needed; returning None.")


if __name__ == "__main__":
    # Run the main coroutine and then shut down async generators explicitly.
    try:
        asyncio.run(main())
    finally:
        # Ensure async generators are properly shut down to avoid errors on loop close.
        with suppress(Exception):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(loop.shutdown_asyncgens())
