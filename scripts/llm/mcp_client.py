# File: mcp_client.py (Corrected dummy exceptions)

import sys
import logging
from pathlib import Path
# Ensure MCP imports are correct
try:
    from mcp import ClientSession, types
    from mcp.client.stdio import stdio_client, StdioServerParameters
    # Import specific exceptions if needed for more granular handling
    from mcp import exceptions as mcp_exceptions
    mcp_available = True
except ImportError:
    logging.critical("MCP library not found or could not be imported. MCP functionality disabled.")
    mcp_available = False
    # Define dummies if necessary to prevent NameErrors downstream

    # --- CORRECTED DUMMY DEFINITIONS ---
    class mcp_exceptions:
        class ToolNotFound(Exception):
            """Dummy exception if mcp is not installed."""
            pass
        class ProtocolError(Exception):
            """Dummy exception if mcp is not installed."""
            pass
        # Add other mcp exceptions here if they are caught specifically later
        # Example:
        # class InvalidArgumentsError(Exception): pass
    # --- END CORRECTION ---

    # Dummy placeholders for other imported items
    class ClientSession: pass
    class types: pass
    class stdio_client: pass
    class StdioServerParameters: pass


async def query_mcp(query_str: str) -> str:
    """
    Queries the RAG MCP server via stdio.

    Args:
        query_str: The query string to send to the RAG tool.

    Returns:
        The response string from the RAG tool, or an error message.
    """
    if not mcp_available:
        return "[MCP ERROR] MCP library not available."

    # Resolve path to server script relative to this client script's location
    try:
        script_dir = Path(__file__).resolve().parent
        # *** Adjust this path based on where rag_mcp_server.py actually lives ***
        # If mcp_client.py and rag_mcp_server.py are both in the root:
        server_script_path = script_dir / "rag_mcp_server.py"
        # If mcp_client.py is in scripts/tools/ and server is in root:
        # server_script_path = script_dir.parents[1] / "rag_mcp_server.py"

        if not server_script_path.is_file():
            server_script_path_arg = "rag_mcp_server.py" # Relative fallback
            logging.warning(f"Could not find server script at {server_script_path}, using relative path '{server_script_path_arg}'")
        else:
            server_script_path_arg = str(server_script_path)

    except Exception as e:
        logging.error(f"Error resolving server script path: {e}")
        server_script_path_arg = "rag_mcp_server.py" # Fallback

    # Ensure python executable is used
    server_params = StdioServerParameters(command=sys.executable, args=[server_script_path_arg])
    logging.info(f"Starting MCP server subprocess: {sys.executable} {server_script_path_arg}")

    try:
        # Use stdio_client context manager
        async with stdio_client(server_params) as (read, write):
            # Use ClientSession context manager
            async with ClientSession(read, write) as session:
                logging.debug("MCP Client: Initializing session...")
                await session.initialize() # Adapt if your MCP lib needs more args
                logging.debug("MCP Client: Session initialized. Calling tool 'rag_query'...")

                try:
                    result = await session.call_tool("rag_query", arguments={"query": query_str})
                    logging.debug("MCP Client: Tool call successful.")
                    return str(result) if result is not None else "[MCP Info] Tool returned None"

                except mcp_exceptions.ToolNotFound: # Catch specific exception
                     logging.error("MCP Error: Tool 'rag_query' not found on server.")
                     return "[MCP ERROR] Tool 'rag_query' not found on server."
                except mcp_exceptions.ProtocolError as pe: # Catch specific exception
                     logging.error(f"MCP Protocol Error: {pe}", exc_info=True)
                     return f"[MCP PROTOCOL ERROR] {pe}"
                # Add more specific MCP exception handling if needed
                except Exception as tool_e:
                     logging.error(f"MCP Error during tool call: {tool_e}", exc_info=True)
                     return f"[MCP ERROR] {type(tool_e).__name__}: {tool_e}"
            # Session automatically closes here

    except Exception as client_e:
        # Catch errors starting stdio_client or during session setup outside tool call
        logging.error(f"Error setting up MCP stdio client/session: {client_e}", exc_info=True)
        return f"[MCP CLIENT ERROR] {type(client_e).__name__}: {client_e}"