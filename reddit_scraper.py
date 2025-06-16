from typing import List
import os
from utils import *
from typing import List
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from tenacity import (
    retry, 
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime, timedelta


load_dotenv()


two_weeks_ago = datetime.today() - timedelta(days=14) 
two_weeks_ago_str = two_weeks_ago.strftime('%Y-%m-%d')


class MCPOverloadedError(Exception):
    pass


mcp_limiter = AsyncLimiter(1, 15)

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

server_params = StdioServerParameters(
    command="npx",
    env={
        "API_TOKEN": os.getenv("API_TOKEN"),
        "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE"),
    },
    args=["@brightdata/mcp"],
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=15, max=60),
    retry=retry_if_exception_type(MCPOverloadedError),
    reraise=True
)
async def process_topic(agent, topic: str):
    async with mcp_limiter:
        messages = [
            {
                "role": "system",
                "content": f"""You are a Reddit analysis expert. Use available tools to:
                1. Find top 2 posts about the given topic BUT only after {two_weeks_ago_str}, NOTHING before this date strictly!
                2. Analyze their content and sentiment
                3. Create a summary of discussions and overall sentiment"""
            },
            {
                "role": "user",
                "content": f"""Analyze Reddit posts about '{topic}'. 
                Provide a comprehensive summary including:
                - Main discussion points
                - Key opinions expressed
                - Any notable trends or patterns
                - Summarize the overall narrative, discussion points and also quote interesting comments without mentioning names
                - Overall sentiment (positive/neutral/negative)"""
            }                   
        ]
        
        try:
            response = await agent.ainvoke({"messages": messages})
            return response["messages"][-1].content
        except Exception as e:
            if "Overloaded" in str(e):
                raise MCPOverloadedError("Service overloaded")
            else:
                raise



async def scrape_reddit_topics(topics: List[str]) -> dict[str, dict]:
    """Process list of topics and return analysis results"""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)
            
            reddit_results = {}
            for topic in topics:
                summary = await process_topic(agent, topic)
                reddit_results[topic] = summary
                await asyncio.sleep(5)  # Maintain rate limiting
                
            return {"reddit_analysis": reddit_results}
