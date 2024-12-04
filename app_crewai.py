if st.sidebar.button("Ask Question"):
    if additional_question.strip() == "":
        st.warning("Please enter a question to proceed.")
    else:
        try:
            # Modify the agent's goal by appending the user's question
            original_goal = agents[selected_agent_for_question].goal
            modified_goal = f"{original_goal} Additionally, {additional_question}"
            
            # Create a temporary agent with the modified goal
            temp_agent = Agent(
                llm=agents[selected_agent_for_question].llm,
                role=agents[selected_agent_for_question].role,
                goal=modified_goal,
                backstory=agents[selected_agent_for_question].backstory,
                tools=agents[selected_agent_for_question].tools,
                allow_delegation=agents[selected_agent_for_question].allow_delegation,
                verbose=agents[selected_agent_for_question].verbose,
            )
            
            # Execute the agent with the modified goal
            crew = Crew(
                agents=[temp_agent],
                tasks=[Task(
                    description=agents[selected_agent_for_question].goal,
                    expected_output=tasks["Advanced Market Research for Premium Locations"].expected_output,  # Adjust as needed
                    output_file="temp_output.txt",
                    agent=temp_agent,
                )],
                verbose=1
            )
            
            st.write("Processing your question with the agent...")
            response = crew.kickoff()
            st.success("Question processed successfully!")
            for task_name, output in response.items():
                st.write(f"**{task_name} Output:**")
                st.write(output)
        except Exception as e:
            st.error(f"An error occurred while processing your question: {e}")
