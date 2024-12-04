import streamlit as st
import os
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI as OpenAI_LLM
from chromadb.config import Settings

# Streamlit App Setup
st.set_page_config(page_title="CrewAI App", layout="wide")

# SQLite workaround: Use DuckDB
settings = Settings(
    chroma_db_impl="duckdb+parquet",  # Avoid SQLite dependency
    persist_directory=".chroma"  # Local directory for saving vectors
)

# User Inputs for API Keys
st.sidebar.header("API Key Setup")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
serper_key = st.sidebar.text_input("SerperDev API Key", type="password")

if not openai_key or not serper_key:
    st.error("Please provide both API keys in the sidebar to proceed.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_key
os.environ["SERPER_API_KEY"] = serper_key

# Cost Options
st.sidebar.header("Choose Model")
model_choice = st.sidebar.radio(
    "Choose the LLM Model:",
    ["Cheaper Option (GPT-3.5)", "Costlier Option (GPT-4)"],
    index=0
)
selected_model = "gpt-3.5-turbo" if model_choice == "Cheaper Option (GPT-3.5)" else "gpt-4"

# Create LLM
parameters = {"temperature": 0.2, "max_tokens": 300}
llm = OpenAI_LLM(model=selected_model, params=parameters)

# Set up Tools
search_tool = SerperDevTool(api_key=serper_key)

# Define Agents
agents = {
    "Real Estate Research Agent": Agent(
        llm=llm,
        role="Senior Real Estate Researcher",
        goal="Find promising properties near water bodies in Trivandrum district.",
        backstory="Veteran real estate expert with 50 years of experience in Trivandrum.",
        tools=[search_tool],
        verbose=1
    ),
    "Furniture Storytelling Agent": Agent(
        llm=llm,
        role="Furniture Storyteller",
        goal="Create engaging stories for handcrafted Kerala furniture for NRIs.",
        backstory="Specialist in Kerala's cultural history and storytelling.",
        verbose=1
    ),
    "Website Design Insight Agent": Agent(
        llm=llm,
        role="Website Design Consultant",
        goal="Analyze top websites to identify design elements for NRIs.",
        backstory="Digital marketing expert specializing in international user experience.",
        tools=[search_tool],
        verbose=1
    )
}

# Define Tasks
tasks = {
    "Advanced Market Research": Task(
        description="Conduct research for properties near water bodies in Trivandrum.",
        expected_output="List best properties with water-view and provide links.",
        output_file="real_estate_report.txt",
        agent=agents["Real Estate Research Agent"]
    ),
    "Furniture Storytelling": Task(
        description="Write stories for Kerala handcrafted furniture.",
        expected_output="Five engaging 300-word stories.",
        output_file="furniture_stories.txt",
        agent=agents["Furniture Storytelling Agent"]
    ),
    "Website Design Insights": Task(
        description="Analyze websites to identify design trends for NRIs.",
        expected_output="Brief outlining impactful website design features.",
        output_file="website_design_insights.txt",
        agent=agents["Website Design Insight Agent"]
    )
}

# Main App
st.title("CrewAI Task Manager")
st.header("Select an Agent or Task to Run")

# Agent/Task Selection
agent_task_choice = st.selectbox(
    "Choose an action:",
    ["Run a single agent", "Run a task", "Run all tasks"]
)

if agent_task_choice == "Run a single agent":
    selected_agent_name = st.selectbox("Choose an agent:", list(agents.keys()))
    if st.button("Run Agent"):
        with st.spinner(f"Running {selected_agent_name}..."):
            agent = agents[selected_agent_name]
            st.write(f"Agent `{selected_agent_name}` is ready for querying.")
            user_query = st.text_area("Ask your question:", "")
            if user_query:
                response = agent.query(user_query)
                st.write("Response:", response)

elif agent_task_choice == "Run a task":
    selected_task_name = st.selectbox("Choose a task:", list(tasks.keys()))
    if st.button("Run Task"):
        with st.spinner(f"Running {selected_task_name}..."):
            task = tasks[selected_task_name]
            crew = Crew(agents=[task.agent], tasks=[task], verbose=1)
            crew.kickoff()
            st.success(f"Task `{selected_task_name}` completed!")

elif agent_task_choice == "Run all tasks":
    if st.button("Run All Tasks"):
        with st.spinner("Running all tasks..."):
            crew = Crew(agents=list(agents.values()), tasks=list(tasks.values()), verbose=1)
            crew.kickoff()
            st.success("All tasks completed!")

