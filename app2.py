# app_crewai.py

import os
import sys
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# 2. Set environment variables from Streamlit secrets
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("OpenAI API key not found in secrets.")

if "SERPER_API_KEY" in st.secrets:
    os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
else:
    st.error("Serper Dev API key not found in secrets.")

# 3. Set Chroma to use DuckDB to avoid sqlite3 dependency
os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"

# 4. Import pysqlite3 and override the default sqlite3
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    st.warning("pysqlite3 is not installed. Proceeding without overriding sqlite3.")

# 5. Import other libraries after setting up environment and overriding modules
import re
import logging
import pandas as pd
import openai
import requests
import time
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain.chat_models import ChatOpenAI
from io import BytesIO

# 6. Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("crew_output.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def is_valid_url(url, retries=3, delay=2):
    """
    Validate URL with multiple retry attempts.
    """
    for attempt in range(retries):
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            if response.status_code == 200:
                return True
            else:
                logging.warning(f"URL check failed ({response.status_code}): {url}")
        except requests.RequestException as e:
            logging.warning(f"URL attempt {attempt + 1} failed: {e}")
        time.sleep(delay)
    return False

def validate_and_normalize_link(link):
    """
    Try to return a valid link. If invalid, return the original text.
    """
    link = link.strip()
    # If link already starts with http or https, verify directly
    if link.startswith('http://') or link.startswith('https://'):
        if is_valid_url(link):
            return link
        else:
            return link  # Return original text if invalid
    else:
        # If not starting with http(s), try prefixing with https://
        potential_link = 'https://' + link
        if is_valid_url(potential_link):
            return potential_link
        else:
            # If still invalid, return original text
            return link

def extract_properties_from_crew_output(crew_output):
    """
    Robust property extraction with improved error handling.
    """
    try:
        # Attempt to extract 'raw' output; adjust attribute names as necessary
        results_text = str(getattr(crew_output, 'raw', getattr(crew_output, 'result', str(crew_output))))
    except Exception as e:
        logging.error(f"Output extraction error: {e}")
        return []
    
    pattern = r'Title:\s*(.*?)\s*Link:\s*(.*?)\s*Snippet:\s*(.*?)\s*(?=Title:|$)'
    matches = re.findall(pattern, results_text, re.DOTALL | re.MULTILINE)
    
    properties = []
    for match in matches:
        try:
            property_dict = {
                'Property Name': match[0].strip(),
                'Link': validate_and_normalize_link(match[1].strip()),
                'Snippet': match[2].strip(),
                'Price': None,
                'Location': 'Trivandrum'
            }
            
            # Extract Price from Snippet
            price_match = re.search(r'₹\s?([\d,]+)', property_dict['Snippet'])
            if price_match:
                price_str = price_match.group(1).replace(',', '')
                try:
                    property_dict['Price'] = float(price_str)
                except ValueError:
                    property_dict['Price'] = None
            
            properties.append(property_dict)
        except Exception as e:
            logging.warning(f"Property processing error: {e}")
    
    return properties

def save_to_excel(properties, filename='trivandrum_real_estate_properties.xlsx'):
    """
    Save properties to Excel with error handling.
    """
    try:
        df = pd.DataFrame(properties)
        
        output = BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        excel_data = output.getvalue()
        
        logging.info(f"Excel file created: {filename}")
        return df, excel_data
    
    except Exception as e:
        logging.error(f"Excel creation error: {e}")
        return None, None

def create_real_estate_crew(search_params):
    """
    Create CrewAI agents with enhanced, flexible search strategy.
    """
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    serper_api_key = os.environ.get('SERPER_API_KEY')

    if not openai_api_key or not serper_api_key:
        raise ValueError("Missing API keys in environment variables.")

    location = search_params.get('location', 'Trivandrum')
    property_type = search_params.get('property_type', 'Waterfront')
    price_range = search_params.get('price_range', 'Any')
    specific_locations = search_params.get('specific_locations')  # List of locations or None

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=2500
    )

    search = SerperDevTool(api_key=serper_api_key)

    # Modify the goal based on whether specific locations are provided
    if specific_locations:
        locations_str = ', '.join(specific_locations)
        search_goal = f"Find and compile a list of {property_type.lower()} properties for sale in {locations_str}, Trivandrum within the price range {price_range} (amounts in rupees)."
        task_description = f"""
        Search for {property_type.lower()} properties for sale in the following locations within Trivandrum:
        {locations_str}.
        Ensure that properties have water views and are within the price range: {price_range} (amounts in rupees).
        Use reputable real estate platforms and provide verified links.
        Format each property as follows:

        'Title: [Name]
        Link: [Verified Link]
        Snippet: [Description] (price in rupees)'
        """
    else:
        search_goal = f"Find and compile a list of {property_type.lower()} properties for sale in {location} within the price range {price_range} (amounts in rupees)."
        task_description = f"""
        Search for {property_type.lower()} properties for sale in {location}.
        Ensure that properties have water views and are within the price range: {price_range} (amounts in rupees).
        Use reputable real estate platforms and provide verified links.
        Format each property as follows:

        'Title: [Name]
        Link: [Verified Link]
        Snippet: [Description] (price in rupees)'
        """

    real_estate_agent = Agent(
        llm=llm,
        role="Real Estate Data Specialist",
        goal=search_goal,
        backstory=(
            "An experienced real estate analyst adept at gathering and verifying property data from multiple sources."
        ),
        allow_delegation=True,
        tools=[search],
        verbose=True
    )

    research_task = Task(
        description=task_description,
        expected_output="A list of at least 10 verified waterfront properties matching the search criteria.",
        agent=real_estate_agent,
    )

    crew = Crew(
        agents=[real_estate_agent],
        tasks=[research_task],
        verbose=1
    )

    return crew

def run_property_search(search_params):
    """
    Enhanced property search with robust error management.
    """
    try:
        logging.info("Initiating comprehensive property search")
        
        crew = create_real_estate_crew(search_params)
        results = crew.kickoff()
        
        logging.info(f"CrewAI Raw Results: {results}")
        
        with st.expander("📄 Raw Search Results"):
            st.write(results)
        
        properties = extract_properties_from_crew_output(results)
        
        logging.info(f"Properties extracted: {len(properties)}")
        
        if properties:
            df, excel_data = save_to_excel(properties)
            return df, excel_data
        else:
            # Fallback: If no properties found and specific_locations are used, perform individual searches
            if search_params.get('specific_locations'):
                logging.info("No properties found in combined search. Initiating individual searches for each location.")
                individual_properties = []
                for loc in search_params['specific_locations']:
                    individual_search_params = search_params.copy()
                    individual_search_params['specific_locations'] = [loc]
                    crew = create_real_estate_crew(individual_search_params)
                    individual_results = crew.kickoff()
                    logging.info(f"CrewAI Raw Results for {loc}: {individual_results}")
                    extracted = extract_properties_from_crew_output(individual_results)
                    individual_properties.extend(extracted)
                
                if individual_properties:
                    properties = individual_properties
                    logging.info(f"Aggregated Properties extracted: {len(properties)}")
                    df, excel_data = save_to_excel(properties)
                    return df, excel_data
                else:
                    logging.warning("No properties found in individual searches as well.")
                    return None, None
            else:
                logging.warning("No properties discovered in search results")
                return None, None
    
    except Exception as e:
        logging.error(f"Comprehensive search failed: {e}", exc_info=True)
        return None, None

def main():
    st.set_page_config(page_title="Trivandrum Real Estate Intelligence", layout="wide")
    st.title("🏘️ Trivandrum Real Estate Intelligence Platform")

    st.sidebar.header("🔍 Property Search Parameters")

    # Step 1: Add search method selection with three options
    search_scope = st.sidebar.radio(
        "Choose Search Scope:",
        ("Trivandrum as a Whole", "All 4 Locations Combined", "Individual Locations")
    )

    # Define the four specific locations
    specific_locations_options = [
        "Kottukal",
        "Kovalam",
        "Thiruvallam",
        "Murukkumpuzha"
    ]

    if search_scope == "Trivandrum as a Whole":
        location = "Trivandrum"
        specific_locations = None
    elif search_scope == "All 4 Locations Combined":
        location = "Trivandrum"
        specific_locations = specific_locations_options  # List of all four locations
    else:  # "Individual Locations"
        selected_locations = st.sidebar.multiselect(
            "Select Locations",
            specific_locations_options,
            default=[specific_locations_options[0]]
        )
        if selected_locations:
            location = "Trivandrum"
            specific_locations = selected_locations
        else:
            st.warning("⚠️ Please select at least one location.")
            location = "Trivandrum"
            specific_locations = None

    property_type = st.sidebar.selectbox(
        "Property Type", 
        ["Waterfront", "Apartment", "Villa", "Commercial", "Land"]
    )
    price_range = st.sidebar.selectbox(
        "Price Range (in ₹)",
        [
            "Any", 
            "₹0 - ₹10,00,000", 
            "₹10,00,001 - ₹50,00,000", 
            "₹50,00,001 - ₹1,00,00,000", 
            "₹1,00,00,001 - ₹2,00,00,000", 
            "₹2,00,00,001 - ₹5,00,00,000", 
            "₹5,00,00,001 - ₹10,00,00,000", 
            "₹10,00,00,001 and above"
        ]
    )

    search_params = {
        'location': location,
        'property_type': property_type,
        'price_range': price_range,
        'specific_locations': specific_locations  # Pass list of specific locations or None
    }

    if 'df' not in st.session_state:
        st.session_state.df = None

    if st.sidebar.button("🔎 Search Properties"):
        with st.spinner("Conducting comprehensive property search..."):
            df, excel_data = run_property_search(search_params)
            
            if df is not None and not df.empty:
                st.session_state.df = df
                st.success(f"✅ Found {len(df)} Properties!")
                
                # Function to create clickable hyperlinks only if URL is valid
                def make_hyperlink(url):
                    url = url.strip()
                    if url.startswith('http://') or url.startswith('https://'):
                        return f'<a href="{url}" target="_blank">{url}</a>'
                    else:
                        return url

                with st.expander("📊 Property Details"):
                    display_df = df.copy()
                    display_df['Property Link'] = display_df['Link'].apply(make_hyperlink)
                    display_df = display_df.drop(columns=['Link'])

                    cols = ['Property Name', 'Property Link', 'Location', 'Price', 'Snippet']
                    cols = [col for col in cols if col in display_df.columns]
                    display_df = display_df[cols]

                    html_table = display_df.to_html(escape=False, index=False)
                    st.markdown(html_table, unsafe_allow_html=True)
                
                st.download_button(
                    label="📥 Download Property Data",
                    data=excel_data,
                    file_name='trivandrum_real_estate_properties.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            else:
                st.warning("⚠️ No properties found. Adjust search parameters.")

if __name__ == "__main__":
    main()
