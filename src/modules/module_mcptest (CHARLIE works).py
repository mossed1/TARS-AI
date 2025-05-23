import asyncio
import json
import logging
import os
import shutil
from typing import Dict, List, Any, Optional

import requests
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global storage for server tools
ALL_TOOLS: Dict[str, List["Tool"]] = {}

class Configuration:
    """Manages configuration and environment variables for the MCP client."""
    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.getenv("XYZ_API_KEY", "")

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from JSON file."""
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        return self.api_key

class Tool:
    """Represents a tool with its properties and formatting."""
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def format_for_llm(self) -> str:
        parts = []
        props = self.input_schema.get('properties', {})
        required = set(self.input_schema.get('required', []))
        for param, info in props.items():
            desc = info.get('description', '')
            req = ' (required)' if param in required else ''
            parts.append(f"- {param}: {desc}{req}")
        return (
            f"Tool: {self.name}\n"
            f"Description: {self.description}\n"
            "Arguments:\n" +
            "\n".join(parts)
        )

class Server:
    """Manages MCP server session and tools."""
    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name = name
        self.config = config
        self.session: Optional[ClientSession] = None
        self.stdio_context: Optional[Any] = None
        self.capabilities: Optional[Dict[str, Any]] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Attach to an existing MCP server and initialize session."""
        cmd = shutil.which('npx') if self.config['command'] == 'npx' else self.config['command']
        params = StdioServerParameters(
            command=cmd,
            args=self.config.get('args', []),
            env={**os.environ, **self.config.get('env', {})}
        )
        # enter stdio client context on this task
        self.stdio_context = stdio_client(params)
        read, write = await self.stdio_context.__aenter__()
        self.session = await ClientSession(read, write).__aenter__()
        self.capabilities = await self.session.initialize()
        #logging.info(f"Server '{self.name}' initialized")

    async def list_tools(self) -> List[Tool]:
        """Return list of tools from this server."""
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        raw = await self.session.list_tools()
        tools: List[Tool] = []
        for item in raw:
            if isinstance(item, tuple) and item[0] == 'tools':
                for t in item[1]:
                    tools.append(Tool(t.name, t.description, t.inputSchema))
        return tools

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on this server."""
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        return await self.session.call_tool(tool_name, arguments)

    async def cleanup(self) -> None:
        """Clean up session and stdio contexts, ignoring cancel-scope errors."""
        async with self._lock:
            if self.session:
                try:
                    await self.session.__aexit__(None, None, None)
                except RuntimeError as e:
                    if "cancel scope" not in str(e):
                        logging.warning(f"Session cleanup error for {self.name}: {e}")
                except Exception as e:
                    logging.warning(f"Session cleanup error for {self.name}: {e}")
                finally:
                    self.session = None

            if self.stdio_context:
                try:
                    await self.stdio_context.__aexit__(None, None, None)
                except RuntimeError as e:
                    if "cancel scope" not in str(e):
                        logging.warning(f"Stdio cleanup error for {self.name}: {e}")
                except Exception as e:
                    logging.warning(f"Stdio cleanup error for {self.name}: {e}")
                finally:
                    self.stdio_context = None

class LLMClient:
    """Handles LLM HTTP requests."""
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        url = "http://192.168.2.59:1234/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": "llama-3.2-90b-vision-preview",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096
        }
        try:
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logging.error(f"LLM request failed: {e}")
            return f"Error: {e}"

class ChatSession:
    """Coordinates user prompts, tool calls, and LLM."""
    def __init__(self, servers: List[Server], llm: LLMClient) -> None:
        self.servers = {s.name: s for s in servers}
        self.llm = llm

    async def process_llm_response(self, response: str) -> str:
        """Detect and dispatch tool calls."""
        try:
            call = json.loads(response)
        except json.JSONDecodeError:
            return response

        tool_name = call.get('tool')
        args = call.get('arguments', {})

        for srv_name, tools in ALL_TOOLS.items():
            if any(t.name == tool_name for t in tools):
                result = await self.servers[srv_name].execute_tool(tool_name, args)
                return f"Tool result: {result}"

        return f"Tool {tool_name} not found"

    async def process_prompt(self, user_prompt: str) -> str:
        # Build system prompt with strong tool-invocation instructions
        tool_descriptions = "\n\n".join(
            t.format_for_llm() for tools in ALL_TOOLS.values() for t in tools
        )
        system_message = f"""You are a helpful assistant with access to these tools:

{tool_descriptions}

IMPORTANT: When you need to use a tool, you must reply ONLY with the exact JSON object format below and nothing else:
{{
  "tool": "tool-name",
  "arguments": {{
    "argument-name": "value"
  }}
}}

If no tool is required, reply only with None."""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_prompt}
        ]

        llm_resp = self.llm.get_response(messages)
        
        #logging.info(f"MCP Response: {llm_resp}")
        if llm_resp == "None":
            print("No Tool Needed")

        return await self.process_llm_response(llm_resp)

async def main() -> None:
    # Load configuration from project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config-MCP.json')
    conf = Configuration.load_config(config_path)

    llm_client = LLMClient(Configuration().llm_api_key)

    # Bootstrap servers and cache their tools
    servers: List[Server] = []
    for name, srv_conf in conf.get('mcpServers', {}).items():
        srv = Server(name, srv_conf)
        await srv.initialize()
        tools = await srv.list_tools()
        ALL_TOOLS[name] = tools
        logging.info(f"Loaded {len(tools)} tools from '{name}'")
        servers.append(srv)

    chat = ChatSession(servers, llm_client)
    try:
        # Example invocation
        resp = await chat.process_prompt("use plywright and describe jovlabs.com/TKD/")
        print(resp)
    finally:
        # Clean shutdown
        for srv in servers:
            try:
                await srv.cleanup()
            except BaseException:
                pass

if __name__ == "__main__":
    asyncio.run(main())
