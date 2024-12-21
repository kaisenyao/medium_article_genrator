import os
from apikey import apikey

from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

os.environ["OPENAI_API_KEY"] = apikey

# we set temperature to 0 because we want an objective research tool without hallucinations
llm = OpenAI(temperature=0.0)

tools = load_tools(["wikipedia", "llm-math"], llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

prompt = input("Input Wikipedia Research Task\n")
agent.run(prompt)
