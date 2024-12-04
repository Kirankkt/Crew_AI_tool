# app_crewai.py

# 1. Override sqlite3 before importing any other modules
import pysqlite3
import sys

# Override the default sqlite3 with pysqlite3
sys.modules['sqlite3'] = pysqlite3

# 2. Now, import the rest of your modules
import streamlit as st
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import pandas as pd
from io import BytesIO
import json
import warnings
import os

# 3. Suppress SyntaxWarnings from pysbd
warnings.filterwarnings("ignore", category=SyntaxWarning)

# 4. Verify sqlite3 version
import sqlite3
st.write(f"**SQLite version:** {sqlite3.sqlite_version}")

# 5. Streamlit UI Setup
st.title("CrewAI Task Manager")
st.sidebar.title("Configuration")

# 6. API Key Inputs
openai_key = st.sidebar.text_input("OpenAI API Key:", type="password")
serper_key = st.sidebar.text_input("Serper API Key:", type="password")

if openai_key and serper_key:
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["SERPER_API_KEY"] = serper_key

    # 7. Model Selection
    model_choice = st.sidebar.radio(
        "Choose a model:",
        ("Cheaper option (GPT-3.5)", "Costlier option (GPT-4)")
    )

    # 8. LLM Setup
    model = "gpt-3.5-turbo" if model_choice == "Cheaper option (GPT-3.5)" else "gpt-4"
    llm = ChatOpenAI(
        model_name=model,
        temperature=0.2,
        max_tokens=300
    )

    # 9. SerperDevTool Setup
    search = SerperDevTool(api_key=serper_key)

    # 10. Define Agents
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

    # 11. Define Tasks
    tasks = {
        "Advanced Market Research for Premium Locations": Task(
            description=(
                "Conduct advanced research for properties near sea, beach, lake, or river, with water-view or sea-view in or around Trivandrum district."
            ),
            expected_output=(
                "A Pandas DataFrame containing property details such as name, price, location, and link."
            ),
            output_file="advanced_market_research_report.xlsx",
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

    # 12. Sidebar: Run Selected Tasks
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
            try:
                with st.spinner("Executing selected tasks..."):
                    crew_output = crew.kickoff()

                st.success("Tasks completed successfully!")

                # 13. Handle CrewOutput properly
                # Access individual task outputs
                for task_output in crew_output.tasks_output:
                    task_name = task_output.task_name
                    output = task_output.output

                    st.write(f"### {task_name} Output:")

                    # Check if output is a Pandas DataFrame
                    if isinstance(output, pd.DataFrame):
                        st.dataframe(output)

                        # Convert DataFrame to Excel in memory
                        excel_buffer = BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            output.to_excel(writer, index=False, sheet_name='Sheet1')
                            writer.save()
                        excel_buffer.seek(0)

                        # Provide download button for Excel file
                        st.download_button(
                            label=f"Download {task_name} as Excel",
                            data=excel_buffer,
                            file_name=tasks[task_name].output_file,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    elif isinstance(output, str):
                        st.write(output)

                        # Provide download button for Text file
                        st.download_button(
                            label=f"Download {task_name} as Text",
                            data=output,
                            file_name=tasks[task_name].output_file,
                            mime="text/plain"
                        )
                    else:
                        # For other types, serialize to JSON
                        json_output = json.dumps(output, indent=4)
                        st.json(output)

                        # Provide download button for JSON file
                        st.download_button(
                            label=f"Download {task_name} as JSON",
                            data=json_output,
                            file_name=tasks[task_name].output_file.replace('.txt', '.json').replace('.xlsx', '.json'),
                            mime="application/json"
                        )
            except Exception as e:
                st.error(f"An error occurred while executing tasks: {e}")
        else:
            st.warning("Please select at least one task to run.")

    # 14. Sidebar: Ask Additional Questions
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
                    # Construct the message
                    message = HumanMessage(content=additional_question)

                    # Get response from the agent's LLM
                    response = agents[selected_agent_for_question].llm([message]).content

                    # Display the response in the main area
                    st.write("### Agent's Response:")
                    st.write(response)

                    # Provide download button for the response
                    st.download_button(
                        label=f"Download {selected_agent_for_question} Response as Text",
                        data=response,
                        file_name=f"{selected_agent_for_question.replace(' ', '_')}_response.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"An error occurred while processing your question: {e}")
    else:
        st.info("You can run predefined tasks or choose to ask specific questions to agents.")

else:
    st.warning("Please enter both OpenAI and Serper API keys to proceed.")

