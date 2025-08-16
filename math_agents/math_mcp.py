from mcp.server.fastmcp import FastMCP
import math_tools

mcp = FastMCP("解数学题的MCP服务", "1.0.0")

mcp.add_tool(math_tools.add, "add", "Adds two integers and returns the result.")

@mcp.tool()
def foo():
    """这是一个愚蠢的工具，什么都不做。"""
    return ""

def main():
    mcp.run('stdio')

if __name__ == "__main__":
    main()