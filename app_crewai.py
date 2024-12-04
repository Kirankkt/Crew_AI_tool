import streamlit as st
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI as OpenAI_LLM
import os

# Streamlit UI Setup
st.title("CrewAI Task Manager")
st.sidebar.title("Configuration")

# API Key Inputs
openai_key = st.sidebar.text_input("OpenAI API Key:", type="password")
serper_key = st.sidebar.text_input("Serper API Key:", type="password")

if openai_key and serper_key:
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["SERPER_API_KEY"] = serper_key

    # Model Selection
    model_choice = st.sidebar.radio(
        "Choose a model:",
        ("Cheaper option (GPT-3.5)", "Costlier option (GPT-4)")
    )

    # LLM Setup
    model = "gpt-3.5-turbo" if model_choice == "Cheaper option (GPT-3.5)" else "gpt-4"
    llm = OpenAI_LLM(
        model=model,
        params={"temperature": 0.2, "max_tokens": 300}
    )

    # SerperDevTool Setup
    search = SerperDevTool(api_key=serper_key)

    # Define Agents
    agents = {
        "Real Estate Research Agent": Agent(
            llm=llm,
            role="Senior Real Estate Researcher",
            goal="Find promising properties for sale (not rent) near water bodies in Trivandrum district.",
            backstory="Veteran Real Estate Agent with 50 years of experience.",
            tools=[search],
            allow_delegation=False,
            verbose=1,
        ),
        "Furniture Storytelling Agent": Agent(
            llm=llm,
            role="Furniture Storyteller",
            goal="Create engaging stories for handcrafted Kerala furniture, emphasizing cultural heritage.",
            backstory="Specialist in Kerala's history and storytelling.",
            allow_delegation=False,
            verbose=1,
        ),
        "Website Design Insight Agent": Agent(
            llm=llm,
            role="Website Design Consultant",
            goal="Analyze top real estate and furniture websites for NRI appeal.",
            backstory="Digital marketing expert specializing in user experience for global clients.",
            tools=[search],
            allow_delegation=False,
            verbose=1,
        ),
    }

    # Define Tasks
    tasks = {
        "Advanced Market Research for Premium Locations": Task(
            description="Research properties near water bodies in Trivandrum district.",
            expected_output="List of properties with links, descriptions, and prices.",
            output_file="real_estate_report.txt",
            agent=agents["Real Estate Research Agent"],
        ),
        "Furniture Storytelling": Task(
            description="Create cultural heritage stories for 5 Kerala furniture pieces.",
            expected_output="Stories connecting the furniture to Kerala's cultural history.",
            output_file="furniture_stories.txt",
            agent=agents["Furniture Storytelling Agent"],
        ),
        "Website Design Insights": Task(
            description="Analyze design elements from 30 websites appealing to NRIs.",
            expected_output="Brief on design features to incorporate in our website.",
            output_file="website_design_insights.txt",
            agent=agents["Website Design Insight Agent"],
        ),
    }

    # Agent Selection
    st.sidebar.header("Select Agents")
    selected_agents = st.sidebar.multiselect(
        "Which agents do you want to run?",
        options=list(agents.keys()),
        default=list(agents.keys())
    )

    # Task Execution
    if st.button("Run Selected Tasks"):
        selected_tasks = [tasks[task_name] for task_name, agent in agents.items() if agent in selected_agents]

        if selected_tasks:
            crew = Crew(
                agents=[agents[name] for name in selected_agents],
                tasks=selected_tasks,
                verbose=1
            )
            st.write("Executing tasks...")
            results = crew.kickoff()
            st.success("Tasks completed!")
            for name, output in results.items():
                st.write(f"**{name} Output:**")
                st.write(output)
        else:
            st.warning("Please select at least one agent to run.")

    # Additional Questions
    st.sidebar.header("Ask Additional Questions")
    additional_question = st.sidebar.text_input("Type your question here:")

    if additional_question and st.button("Ask Question"):
        selected_agent = st.sidebar.selectbox(
            "Which agent should answer your question?",
            options=list(agents.keys())
        )
        response = agents[selected_agent].llm.chat(additional_question)
        st.write("**Agent's Response:**")
        st.write(response)

else:
    st.warning("Please enter both OpenAI and Serper API keys to proceed.")
