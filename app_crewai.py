import pysqlite3
import sys
import streamlit as st
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os
import warnings

# Suppress SyntaxWarnings from pysbd
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Override the default sqlite3 with pysqlite3
sys.modules['sqlite3'] = pysqlite3

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
    llm = ChatOpenAI(
        model_name=model,
        temperature=0.2,
        max_tokens=300
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
        "Financial Reporting Agent": Agent(
            llm=llm,
            role="Financial Manager",
            goal="Calculate monthly expenses, profitability, and cost analysis for the business.",
            backstory="An accounting genius with a knack for analyzing financial data.",
            allow_delegation=False,
            verbose=1,
        ),
    }

    # Define Tasks
    tasks = {
        "Advanced Market Research for Premium Locations": Task(
            description=(
                "Conduct advanced research for properties near sea, beach, lake, or river, with water-view or sea-view in or around Trivandrum district."
            ),
            expected_output=(
                "List all the best properties (with accurate links for each of them) that are near sea, beach, lake, or river, with water-view or sea-view in or around Trivandrum district based on your search."
            ),
            output_file="advanced_market_research_report.txt",
            agent=agents["Real Estate Research Agent"],
        ),
        "Furniture Storytelling": Task(
            description=(
                "Write 5 unique stories for handcrafted Kerala furniture pieces that emphasize cultural heritage and craftsmanship. "
                "Each story should highlight the furniture's design, craftsmanship, and cultural background, appealing to NRIs."
            ),
            expected_output=(
                "Each story should be approximately 300 words, maintaining a consistent brand voice and connecting the furniture piece to Kerala’s rich cultural heritage."
            ),
            output_file="furniture_stories.txt",
            agent=agents["Furniture Storytelling Agent"],
        ),
        "Website Design Insights": Task(
            description=(
                "Analyze 30 inspiring real estate and furniture websites to identify design elements that appeal to NRIs. "
                "Focus on layout, color schemes, user experience, and interactive features."
            ),
            expected_output=(
                "A design brief outlining the most impactful features, layouts, and user experience elements to incorporate into our luxury real estate and furniture website."
            ),
            output_file="website_design_insights.txt",
            agent=agents["Website Design Insight Agent"],
        ),
        "Financial Report Calculation": Task(
            description="Compile all expenses and calculate profitability for monthly financial reporting.",
            expected_output="An Excel report detailing costs and profitability.",
            output_file="financial_report.xlsx",
            agent=agents["Financial Reporting Agent"],
        ),
    }

    # Sidebar: Run Selected Tasks
    st.sidebar.header("Run Selected Tasks")
    selected_tasks = st.sidebar.multiselect(
        "Select tasks to run:",
        options=list(tasks.keys()),
        default=list(tasks.keys())
    )

    if st.sidebar.button("Run Selected Tasks"):
        if selected_tasks:
            selected_task_objects = [tasks[task_name] for task_name in selected_tasks]
            crew = Crew(
                agents=[task.agent for task in selected_task_objects],
                tasks=selected_task_objects,
                verbose=1
            )
            st.write("Executing selected tasks...")
            try:
                results = crew.kickoff()
                st.success("Tasks completed successfully!")
                for task_name, output in results.items():
                    st.write(f"**{task_name} Output:**")
                    st.write(output)
            except Exception as e:
                st.error(f"An error occurred while executing tasks: {e}")
        else:
            st.warning("Please select at least one task to run.")

    # Sidebar: Ask Additional Questions
    st.sidebar.header("Ask Additional Questions")
    ask_question = st.sidebar.checkbox("Do you want to ask a specific question to an agent?")
    
    if ask_question:
        selected_agent_for_question = st.sidebar.selectbox(
            "Select an agent to ask a question:",
            options=list(agents.keys())
        )
        additional_question = st.sidebar.text_input("Type your question here:")

        if st.sidebar.button("Ask Question"):
            if additional_question.strip() == "":
                st.warning("Please enter a question to proceed.")
            else:
                try:
                    # Integrate the additional question into the agent's context
                    # This can be done by appending the question to the agent's goal or sending it as a separate prompt
                    # Here, we'll send it as a separate prompt for simplicity

                    # Construct the message
                    message = HumanMessage(content=additional_question)
                    
                    # Get response from the agent's LLM
                    response = agents[selected_agent_for_question].llm([message]).content

                    st.write("**Agent's Response:**")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred while processing your question: {e}")

    else:
        st.info("You can run predefined tasks or choose to ask specific questions to agents.")

else:
    st.warning("Please enter both OpenAI and Serper API keys to proceed.")
