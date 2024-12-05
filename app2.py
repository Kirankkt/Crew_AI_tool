import os
import re
import logging
import pandas as pd
import openai
import requests
import time
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI as OpenAI_LLM
import streamlit as st
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    filemode='w'
)

def is_valid_url(url, retries=3, delay=2):
    """
    Check if a URL is valid by sending a HEAD request.
    Retries a specified number of times in case of transient failures.
    """
    for attempt in range(retries):
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            if response.status_code == 200:
                # Optionally, check for specific content in the page to ensure it's a property listing
                return True
            else:
                logging.warning(f"URL check failed ({response.status_code}): {url}")
        except requests.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed for URL: {url} with error: {e}")
        time.sleep(delay)
    return False

def validate_and_normalize_link(link):
    """
    Validate and normalize property links.
    Ensures the link is properly formatted and points to a valid page.
    """
    # Define URL patterns specific to real estate websites
    url_patterns = [
        r'^https?://www\.magicbricks\.com/property-details/\S+',
        r'^https?://www\.99acres\.com/property/\S+',
        r'^https?://\S+',      # Generic HTTP/HTTPS URLs
        r'^www\.\S+',          # URLs starting with www
        r'^\S+\.(com|in|org|net)\S*'  # URLs containing common TLDs
    ]
    
    # Remove any trailing characters or whitespace
    link = link.strip()
    
    for pattern in url_patterns:
        if re.match(pattern, link, re.IGNORECASE):
            # Ensure the link starts with http:// or https://
            if not link.startswith(('http://', 'https://')):
                link = 'https://' + link
            
            if is_valid_url(link):
                return link
            else:
                logging.warning(f"Invalid URL detected: {link}")
                return f"Invalid Property Link: {link}"
    
    # If no valid link found, return a meaningful placeholder
    return f"Property Link Not Available: {link}" if link else "No Link Provided"

def extract_properties_from_crew_output(crew_output):
    """
    Extract properties from CrewAI output object with improved link handling.
    """
    try:
        # Try different methods to extract string content
        if hasattr(crew_output, 'raw'):
            results_text = str(crew_output.raw)
        elif hasattr(crew_output, 'result'):
            results_text = str(crew_output.result)
        else:
            results_text = str(crew_output)
    except Exception as e:
        logging.error(f"Error converting output to string: {e}")
        return []

    # Regex pattern to extract property details
    pattern = r'(\d+)\.\s*Property Name:\s*(.*?)\s*Location:\s*(.*?)\s*Price:\s*(.*?)\s*Water View Type:\s*(.*?)\s*Contact Information:\s*(.*?)\s*Property Link:\s*(.*?)(?=\d+\.\s*|$)'

    # Find all matches
    matches = re.findall(pattern, results_text, re.DOTALL | re.MULTILINE)

    # Convert to list of dictionaries
    properties = []
    for match in matches:
        try:
            property_dict = {
                'Property Number': match[0],
                'Property Name': match[1].strip(),
                'Location': match[2].strip(),
                'Price': match[3].strip(),
                'Water View Type': match[4].strip(),
                'Contact Information': match[5].strip(),
                'Property Link': validate_and_normalize_link(match[6].strip())
            }
            properties.append(property_dict)
        except Exception as e:
            logging.warning(f"Error processing property: {e}")
    
    return properties

def save_to_excel(properties, filename='trivandrum_real_estate_properties.xlsx'):
    """
    Save properties to an Excel file.
    """
    try:
        # Create DataFrame
        df = pd.DataFrame(properties)
        
        # Save to Excel using a BytesIO buffer
        output = BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        excel_data = output.getvalue()
        
        logging.info(f"Excel file successfully created: {filename}")
        return df, excel_data
    
    except Exception as e:
        logging.error(f"Error creating Excel file: {e}")
        return None, None

def create_real_estate_crew(openai_api_key, serper_api_key, search_params):
    """
    Create and configure the CrewAI agents and tasks with dynamic search parameters.
    """
    # Extract search parameters
    location = search_params.get('location', 'Trivandrum')
    property_type = search_params.get('property_type', 'waterfront')
    price_range = search_params.get('price_range', 'any')

    # Create the primary LLM
    llm = OpenAI_LLM(
        openai_api_key=openai_api_key,
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=1500
    )

    # SerperDevTool setup for web search
    search = SerperDevTool(api_key=serper_api_key)

    # Define the real estate research agent
    real_estate_agent = Agent(
        llm=llm,
        role="Senior Real Estate Researcher",
        goal=f"Find promising {property_type} properties for sale in {location} district near water bodies",
        backstory=(
            "A veteran Real Estate Agent with 50 years of experience in Trivandrum real estate, "
            "specializing in identifying unique waterfront properties. Known for meticulous research "
            "and ability to uncover hidden property gems near water bodies."
        ),
        allow_delegation=True,
        tools=[search],
        verbose=True
    )

    # Define the research task with enhanced instructions
    description = (
        f"Conduct a comprehensive search for {property_type} properties near water bodies in {location} district. "
        "Focus on properties with sea, beach, lake, or river views that are currently for sale. "
        f"Price range: {price_range}. "
        "\n\nCRITICAL REQUIREMENTS:"
        "\n- Prioritize properties actually for SALE, not rentals"
        "\n- Include VERIFIED property links from reputable sources"
        "\n- Use local real estate websites like MagicBricks, 99acres, or local Trivandrum portals"
        "\n- If no direct link is available, provide most relevant contact information"
        "\n\nOutput Format (EXACTLY):"
        "\n'1. Property Name: [Name] Location: [Location] Price: [Price] "
        "Water View Type: [View Type] Contact Information: [Contact] "
        "Property Link: [VERIFIED LINK]'"
    )

    research_task = Task(
        description=description,
        expected_output=(
            "Detailed list of 3-5 waterfront properties with complete, verifiable information. "
            "Each property must have a comprehensive description and a valid link."
        ),
        agent=real_estate_agent,
    )

    # Assemble the crew
    crew = Crew(
        agents=[real_estate_agent],
        tasks=[research_task],
        verbose=1
    )

    return crew

def run_property_search(openai_api_key, serper_api_key, search_params):
    """
    Main function to run the property search.
    """
    try:
        # Log the start of the search
        logging.info("Starting Trivandrum waterfront property search")
        
        # Create and run the crew
        crew = create_real_estate_crew(openai_api_key, serper_api_key, search_params)
        results = crew.kickoff()
        
        # Extract properties from CrewAI output
        properties = extract_properties_from_crew_output(results)
        
        # Log the number of properties found
        logging.info(f"Found {len(properties)} properties")
        
        if properties:
            df, excel_data = save_to_excel(properties)
            return df, excel_data
        else:
            logging.warning("No properties found in the results")
            return None, None
    
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return None, None

def handle_user_query(query, openai_api_key, df):
    """
    Handle dynamic user queries based on the current property data.
    """
    if df is None or df.empty:
        return "No property data available. Please perform a search first."

    # Create a prompt based on the user's question and the data
    prompt = f"""
    You are a real estate assistant. Here are the properties found:

    {df.to_string(index=False)}

    Based on the above data, answer the following question:

    {query}
    """

    # Use OpenAI API to get the answer
    try:
        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful real estate assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        return response.choices[0].message['content']
    except Exception as e:
        logging.error(f"Error handling user query: {e}")
        return "An error occurred while processing your query."

def main():
    st.set_page_config(page_title="Trivandrum Real Estate Assistant", layout="wide")
    st.title("🏠 Trivandrum Real Estate Assistant")

    st.sidebar.header("🔑 API Keys")
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
    serper_api_key = st.sidebar.text_input("Enter your Serper Dev API key:", type="password")

    st.sidebar.header("🔍 Search Parameters")
    location = st.sidebar.text_input("Location", "Trivandrum")
    property_type = st.sidebar.selectbox("Property Type", ["Waterfront", "Apartment", "Villa", "Commercial"])
    price_range = st.sidebar.text_input("Price Range", "Any")

    search_params = {
        'location': location,
        'property_type': property_type,
        'price_range': price_range
    }

    if st.sidebar.button("Search Properties"):
        if openai_api_key and serper_api_key:
            with st.spinner("Searching for properties..."):
                df, excel_data = run_property_search(openai_api_key, serper_api_key, search_params)
                if df is not None:
                    st.success("✅ Properties Found!")
                    st.dataframe(df)
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name='trivandrum_real_estate_properties.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                else:
                    st.warning("⚠️ No properties found. Please try different search parameters.")
        else:
            st.error("❗ Please enter both OpenAI and Serper Dev API keys.")

    st.header("💬 Ask a Question")
    user_query = st.text_input("Your Question:")
    if st.button("Submit Question"):
        if openai_api_key:
            # Load existing property data
            if 'df' in locals() and df is not None:
                answer = handle_user_query(user_query, openai_api_key, df)
                st.write(answer)
            else:
                st.error("❗ No property data available. Please perform a search first.")
        else:
            st.error("❗ Please enter your OpenAI API key.")

if __name__ == "__main__":
    main()
