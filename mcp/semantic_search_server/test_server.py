import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Define the command to start your server.
    # Adjust the path to src/main.py as necessary.
    server_params = StdioServerParameters(
        command="python",
        args=["src/main.py"]
    )

    # Connect to the MCP server using the stdio transport.
    async with stdio_client(server_params) as (read_stream, write_stream):
        # Create a client session.
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the MCP connection.
            await session.initialize()

            # (Optional) List available tools to verify that your tool is registered.
            tools_response = await session.list_tools()
            print("Available tools:", [tool.name for tool in tools_response.tools])

            # Perform a simple request by calling your tool.
            # Here we call the "search_about_neuroconv" tool with sample parameters.
            result = await session.call_tool("search_about_neuroconv", {
                "query": "What is NeuroConv?",
                "context": "Basic introduction"
            })

            # Process and print the output.
            for content in result.content:
                if content.type == "text":
                    print("Tool output:", content.text)

if __name__ == "__main__":
    asyncio.run(main())
