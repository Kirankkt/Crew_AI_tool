import os
import sys
import streamlit as st

# 1. Set environment variables from Streamlit secrets
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("OpenAI API key not found in secrets.")

if "SERPER_DEV_API_KEY" in st.secrets:
    os.environ["SERPER_DEV_API_KEY"] = st.secrets["SERPER_DEV_API_KEY"]
else:
    st.error("Serper Dev API key not found in secrets.")

# 2. Set Chroma to use DuckDB to avoid sqlite3 dependency
os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"

# 3. Import pysqlite3 and override the default sqlite3
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    st.warning("pysqlite3 is not installed. Proceeding without overriding sqlite3.")

# 4. Import other libraries after setting up environment and overriding modules
import re
import logging
import pandas as pd
import openai
import requests
import time
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI as OpenAI_LLM
from io import BytesIO

# 5. Suppress SyntaxWarnings from pysbd (optional)
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    filemode='w'
)

# Temporary debug statements (Remove after verification)
st.write(f"OpenAI API Key Set: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
st.write(f"Serper Dev API Key Set: {'Yes' if os.getenv('SERPER_DEV_API_KEY') else 'No'}")

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
    Comprehensive link validation and normalization.
    """
    url_patterns = [
        r'^https?://www\.magicbricks\.com/property-details/\S+',
        r'^https?://www\.99acres\.com/property/\S+',
        r'^https?://\S+',      # Generic HTTP/HTTPS URLs
        r'^www\.\S+',          # URLs starting with www
        r'^\S+\.(com|in|org|net)\S*'  # URLs containing common TLDs
    ]
    
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
                return f"Unverified Property Link: {link}"
    
    return f"Property Link Not Available: {link}" if link else "No Link Provided"

def extract_properties_from_crew_output(crew_output):
    """
    Robust property extraction with improved error handling.
    """
    try:
        # Multiple ways to extract output string
        results_text = str(getattr(crew_output, 'raw', 
                         getattr(crew_output, 'result', 
                         str(crew_output))))
    except Exception as e:
        logging.error(f"Output extraction error: {e}")
        return []

    # Updated regex based on actual CrewAI output format
    pattern = r'(\d+)\.\s*Property Name:\s*(.*?)\s*Location:\s*(.*?)\s*Price:\s*(.*?)\s*Water View Type:\s*(.*?)\s*Contact Information:\s*(.*?)\s*Property Link:\s*(.*?)(?=\n\d+\.\s*|$)'

    matches = re.findall(pattern, results_text, re.DOTALL | re.MULTILINE)

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
    # Retrieve API keys
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    serper_api_key = os.environ.get('SERPER_DEV_API_KEY')

    if not openai_api_key or not serper_api_key:
        raise ValueError("Missing API keys in environment variables.")

    # Extract search parameters
    location = search_params.get('location', 'Trivandrum')
    property_type = search_params.get('property_type', 'Waterfront')
    price_range = search_params.get('price_range', 'Any')

    # Configure the LLM
    llm = OpenAI_LLM(
        openai_api_key=openai_api_key,
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=2500
    )

    # Configure the SerperDevTool for web search
    search = SerperDevTool(api_key=serper_api_key)

    # Define the real estate research agent
    real_estate_agent = Agent(
        llm=llm,
        role="Real Estate Data Specialist",
        goal=f"Find and compile a list of {property_type.lower()} properties for sale in {location} within the price range {price_range}.",
        backstory=(
            "An experienced real estate analyst adept at gathering and verifying property data from multiple sources."
        ),
        allow_delegation=True,
        tools=[search],
        verbose=True
    )

    # Define the research task with clear instructions
    research_task = Task(
        description=f"""
        Search for {property_type.lower()} properties for sale in {location}.
        Ensure that properties have water views and are within the price range: {price_range}.
        Use reputable real estate platforms and provide verified links.
        Format each property as follows:

        '1. Property Name: [Name]
        Location: [Location]
        Price: [Price]
        Water View Type: [Type]
        Contact Information: [Contact]
        Property Link: [Verified Link]'
        """,
        expected_output="A list of 3-5 verified waterfront properties matching the search criteria.",
        agent=real_estate_agent,
    )

    # Assemble the crew
    crew = Crew(
        agents=[real_estate_agent],
        tasks=[research_task],
        verbose=2
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
        
        # Log and display the raw CrewAI results
        logging.info(f"CrewAI Raw Results: {results}")
        
        with st.expander("📄 Raw Search Results"):
            st.write(results)
        
        properties = extract_properties_from_crew_output(results)
        
        logging.info(f"Properties extracted: {len(properties)}")
        
        if properties:
            df, excel_data = save_to_excel(properties)
            return df, excel_data
        else:
            logging.warning("No properties discovered in search results")
            return None, None
    
    except Exception as e:
        logging.error(f"Comprehensive search failed: {e}", exc_info=True)
        return None, None

def handle_user_query(query, df):
    """
    Advanced query processing with comprehensive data analysis.
    """
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    if not openai_api_key:
        return "OpenAI API key is missing from environment variables."

    if df is None or df.empty:
        return "No property data available. Perform a search first."

    # Advanced query preprocessing
    try:
        # Convert price to numeric, handling potential formatting issues
        df['Numeric_Price'] = df['Price'].replace({
            '₹': '', 
            ',': '', 
            ' ': ''
        }, regex=True).astype(float)

        # Prepare comprehensive context
        context = f"""
        PROPERTY DATASET OVERVIEW:
        - Total Properties: {len(df)}
        - Price Statistics:
          * Minimum Price: ₹{df['Numeric_Price'].min():,.2f}
          * Maximum Price: ₹{df['Numeric_Price'].max():,.2f}
          * Average Price: ₹{df['Numeric_Price'].mean():,.2f}
          * Median Price: ₹{df['Numeric_Price'].median():,.2f}

        Columns: {', '.join(df.columns)}
        """

        preprocessed_query = f"""
        ADVANCED REAL ESTATE DATA ANALYSIS

        {context}

        USER QUERY: {query}

        Provide a comprehensive, data-driven response using the available dataset.
        If direct data is insufficient, explain limitations and provide contextual insights.
        """

        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a precise real estate data analyst."},
                {"role": "user", "content": preprocessed_query}
            ],
            temperature=0.3,
        )
        
        return response.choices[0].message['content']

    except Exception as e:
        logging.error(f"Query processing error: {e}", exc_info=True)
        return f"Query processing failed. Error: {str(e)}"

def main():
    st.set_page_config(page_title="Trivandrum Real Estate Intelligence", layout="wide")
    st.title("🏘️ Trivandrum Real Estate Intelligence Platform")

    # Environment variable key check
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    serper_api_key = os.environ.get('SERPER_DEV_API_KEY')

    if not openai_api_key or not serper_api_key:
        st.error("❌ API keys missing. Set OPENAI_API_KEY and SERPER_DEV_API_KEY in Streamlit secrets.")
        st.stop()

    # Sidebar configuration
    st.sidebar.header("🔍 Property Search Parameters")
    location = st.sidebar.text_input("Location", "Trivandrum")
    property_type = st.sidebar.selectbox(
        "Property Type", 
        ["Waterfront", "Apartment", "Villa", "Commercial", "Land"]
    )
    price_range = st.sidebar.text_input("Price Range", "Any")

    search_params = {
        'location': location,
        'property_type': property_type,
        'price_range': price_range
    }

    # Session state initialization
    if 'df' not in st.session_state:
        st.session_state.df = None

    # Property search section
    if st.sidebar.button("🔎 Search Properties"):
        with st.spinner("Conducting comprehensive property search..."):
            df, excel_data = run_property_search(search_params)
            
            if df is not None and not df.empty:
                st.session_state.df = df
                st.success(f"✅ Found {len(df)} Properties!")
                
                # Expandable property view
                with st.expander("📊 Property Details"):
                    st.dataframe(df)
                
                # Download button
                st.download_button(
                    label="📥 Download Property Data",
                    data=excel_data,
                    file_name='trivandrum_real_estate_properties.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            else:
                st.warning("⚠️ No properties found. Adjust search parameters.")

    # Query section
    st.header("💬 Intelligent Property Insights")
    user_query = st.text_input("Ask a detailed question about the properties")
    
    if st.button("Get Insights"):
        if st.session_state.df is not None:
            with st.spinner("Analyzing property data..."):
                answer = handle_user_query(user_query, st.session_state.df)
                st.write(answer)
        else:
            st.error("❗ Perform a property search first")

    # Optional: Remove temporary debug statements after verification
    # st.write(f"OpenAI API Key Set: {'Yes' if openai_api_key else 'No'}")
    # st.write(f"Serper Dev API Key Set: {'Yes' if serper_api_key else 'No'}")

if __name__ == "__main__":
    main()
