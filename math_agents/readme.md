# Math Agents

这是一套agents，专门用来处理数学相关的任务。

# Setup

在math_agents目录下，创建一个`.env`文件，内容如下：

```bash
OPENAI_API_KEY=你的OpenAI API Key
OPENAI_MODEL=gpt-4o # 或者其他你想使用的模型
```

然后在根目录运行命令

```bash
python math_agents/oepnai_tool.py
```

# 配置MCP

可以用Claude Desktop或者Cline来运行这个MCP。

在Claude Desktop中，打开Settings，找到MCP配置部分。
然后在MCP配置中添加以下内容：

```json
{
    "mcpServers": {
        "Prince of Math": {
            "command": "/Users/daddyyang/.local/bin/uv",
            "args": [
                "--directory",
                "/Users/daddyyang/sourcecode/little_ai_zoo",
                "run",
                "math_agents/math_mcp.py"
            ]
        }
    }
}
```


# Refers
1. https://www.bilibili.com/video/BV1nyVDzaE1x?spm_id_from=333.788.player.switch&vd_source=472fb6e92fd30d1ed74891f42c6b5a38
2. 