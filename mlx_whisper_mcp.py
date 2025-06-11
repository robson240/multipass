# mlx_whisper_mcp.py
"""
Model Context Protocol (MCP) Server for MLX Whisper
Provides a standardized interface for AI assistants to use whisper transcription
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

from mlx_whisper_client import MLXWhisper, WhisperClient

logger = logging.getLogger(__name__)


class WhisperMCPServer:
    """MCP Server implementation for MLX Whisper"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.server = Server("mlx-whisper")
        self.api_url = api_url
        self.client: Optional[WhisperClient] = None
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup MCP protocol handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="transcribe_audio",
                    description="Transcribe audio data to text using Whisper",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "audio_path": {
                                "type": "string",
                                "description": "Path to audio file"
                            },
                            "audio_base64": {
                                "type": "string",
                                "description": "Base64 encoded audio data"
                            },
                            "language": {
                                "type": "string",
                                "description": "Language code (e.g., 'en')",
                                "default": "en"
                            },
                            "model": {
                                "type": "string",
                                "enum": ["tiny", "base", "small", "medium", "large"],
                                "description": "Model size",
                                "default": "base"
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Sampling temperature",
                                "default": 0.0
                            }
                        },
                        "oneOf": [
                            {"required": ["audio_path"]},
                            {"required": ["audio_base64"]}
                        ]
                    }
                ),
                Tool(
                    name="list_whisper_models",
                    description="List available Whisper models",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="check_whisper_health",
                    description="Check Whisper API health status",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
            
        @self.server.call_tool()
        async def handle_call_tool(
            name: str,
            arguments: Optional[Dict[str, Any]] = None
        ) -> List[TextContent]:
            """Handle tool calls"""
            
            if not self.client:
                self.client = WhisperClient(self.api_url)
                
            try:
                if name == "transcribe_audio":
                    result = await self._transcribe_audio(arguments or {})
                elif name == "list_whisper_models":
                    result = await self._list_models()
                elif name == "check_whisper_health":
                    result = await self._check_health()
                else:
                    result = f"Unknown tool: {name}"
                    
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
                
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "tool": name
                    })
                )]
                
        @self.server.list_resources()
        async def handle_list_resources() -> List[str]:
            """List available resources"""
            return [
                "audio://microphone",
                "config://whisper/settings"
            ]
            
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read a resource"""
            if uri == "config://whisper/settings":
                return json.dumps({
                    "api_url": self.api_url,
                    "default_model": "base",
                    "default_language": "en"
                })
            else:
                raise ValueError(f"Unknown resource: {uri}")
                
    async def _transcribe_audio(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe audio using the API"""
        if not self.client:
            self.client = WhisperClient(self.api_url)
            
        # Handle different input types
        if "audio_path" in args:
            # Read audio file
            audio_path = Path(args["audio_path"])
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
            # For simplicity, assuming it's already in the right format
            # In production, you'd use proper audio loading
            import numpy as np
            audio = np.load(audio_path) if audio_path.suffix == '.npy' else None
            
            if audio is None:
                # Try transcribe_file endpoint
                result = await self.client.transcribe_file(str(audio_path))
            else:
                result = await self.client.transcribe(
                    audio,
                    language=args.get("language", "en"),
                    model=args.get("model", "base"),
                    temperature=args.get("temperature", 0.0)
                )
                
        elif "audio_base64" in args:
            # Handle base64 audio
            import base64
            audio_bytes = base64.b64decode(args["audio_base64"])
            audio = np.frombuffer(audio_bytes, dtype=np.float32)
            
            result = await self.client.transcribe(
                audio,
                language=args.get("language", "en"),
                model=args.get("model", "base"),
                temperature=args.get("temperature", 0.0)
            )
        else:
            raise ValueError("Either audio_path or audio_base64 must be provided")
            
        return {
            "text": result.get("text", ""),
            "language": result.get("language"),
            "processing_time": result.get("processing_time"),
            "model_used": result.get("model_used")
        }
        
    async def _list_models(self) -> Dict[str, Any]:
        """List available models"""
        if not self.client:
            self.client = WhisperClient(self.api_url)
            
        models = await self.client.list_models()
        return {
            "models": models,
            "recommended": "base",
            "fastest": "tiny",
            "most_accurate": "large"
        }
        
    async def _check_health(self) -> Dict[str, Any]:
        """Check API health"""
        if not self.client:
            self.client = WhisperClient(self.api_url)
            
        health = await self.client.health()
        return {
            "status": health.get("status"),
            "timestamp": health.get("timestamp"),
            "api_url": self.api_url,
            "services_available": len(health.get("services", {}))
        }
        
    async def run(self):
        """Run the MCP server"""
        # Initialize with options
        init_options = InitializationOptions(
            server_name="mlx-whisper",
            server_version="1.0.0",
            capabilities=self.server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={}
            )
        )
        
        # Run server
        async with self.client or WhisperClient(self.api_url) as client:
            self.client = client
            await self.server.run(
                init_options=init_options,
                # Use stdio transport by default
                transport="stdio"
            )


# ============= Advanced MCP Features =============

class AdvancedWhisperMCP(WhisperMCPServer):
    """Extended MCP server with advanced features"""
    
    def setup_handlers(self):
        """Setup extended handlers"""
        super().setup_handlers()
        
        @self.server.list_prompts()
        async def handle_list_prompts() -> List[Dict[str, Any]]:
            """List available prompts"""
            return [
                {
                    "name": "transcribe_meeting",
                    "description": "Transcribe and summarize a meeting recording",
                    "arguments": [
                        {
                            "name": "audio_file",
                            "description": "Path to meeting audio",
                            "required": True
                        },
                        {
                            "name": "attendees",
                            "description": "List of attendees",
                            "required": False
                        }
                    ]
                },
                {
                    "name": "extract_action_items",
                    "description": "Extract action items from transcription",
                    "arguments": [
                        {
                            "name": "transcription",
                            "description": "Meeting transcription text",
                            "required": True
                        }
                    ]
                }
            ]
            
        @self.server.get_prompt()
        async def handle_get_prompt(
            name: str,
            arguments: Optional[Dict[str, str]] = None
        ) -> List[Any]:
            """Get a specific prompt"""
            
            if name == "transcribe_meeting":
                audio_file = arguments.get("audio_file", "")
                attendees = arguments.get("attendees", "Unknown")
                
                return [
                    TextContent(
                        type="text",
                        text=f"Please transcribe the meeting audio from {audio_file}. "
                        f"Attendees: {attendees}. Provide a summary with key points, "
                        "decisions made, and action items."
                    )
                ]
            elif name == "extract_action_items":
                transcription = arguments.get("transcription", "")
                
                return [
                    TextContent(
                        type="text",
                        text=f"Extract all action items from this transcription:\n\n"
                        f"{transcription}\n\n"
                        "Format as a list with assignee, task, and due date if mentioned."
                    )
                ]
            else:
                raise ValueError(f"Unknown prompt: {name}")
                
        @self.server.list_resource_templates()
        async def handle_list_resource_templates() -> List[Dict[str, Any]]:
            """List resource templates"""
            return [
                {
                    "uriTemplate": "audio://file/{path}",
                    "name": "Audio files",
                    "description": "Access audio files for transcription",
                    "mimeType": "audio/*"
                },
                {
                    "uriTemplate": "transcript://session/{id}",
                    "name": "Transcription sessions",
                    "description": "Access previous transcription sessions",
                    "mimeType": "application/json"
                }
            ]
            
        @self.server.complete()
        async def handle_complete(
            ref: Dict[str, Any],
            argument: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Handle completions"""
            
            if ref.get("type") == "resource" and "audio://file/" in ref.get("uri", ""):
                # Complete audio file paths
                import glob
                pattern = argument.get("value", "") + "*"
                files = glob.glob(pattern)
                
                return {
                    "completion": [
                        {"value": f, "label": Path(f).name}
                        for f in files[:10]  # Limit results
                    ]
                }
                
            return {"completion": []}


# ============= CLI and Testing =============

async def test_mcp_server():
    """Test the MCP server"""
    server = AdvancedWhisperMCP()
    
    # Simulate tool calls
    print("Testing MCP Server...")
    
    # List tools
    tools = await server.server.list_tools()
    print(f"\nAvailable tools: {[t.name for t in tools]}")
    
    # Check health
    result = await server._check_health()
    print(f"\nHealth check: {result}")
    
    # List models
    models = await server._list_models()
    print(f"\nAvailable models: {models}")
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MLX Whisper MCP Server")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="MLX Whisper API URL")
    parser.add_argument("--test", action="store_true",
                        help="Run tests instead of server")
    parser.add_argument("--advanced", action="store_true",
                        help="Use advanced MCP features")
    
    args = parser.parse_args()
    
    if args.test:
        asyncio.run(test_mcp_server())
    else:
        # Run MCP server
        ServerClass = AdvancedWhisperMCP if args.advanced else WhisperMCPServer
        server = ServerClass(args.api_url)
        
        try:
            asyncio.run(server.run())
        except KeyboardInterrupt:
            print("\nShutting down MCP server...")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
