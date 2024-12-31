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

def create_real_estate_crew(search_params, latitudes=None, longitudes=None):
    """
    Create CrewAI agents for specific waterfront property search in predefined locations.
    """
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    serper_api_key = os.environ.get('SERPER_API_KEY')

    if not openai_api_key or not serper_api_key:
        raise ValueError("Missing API keys in environment variables.")

    location = search_params.get('location', 'Trivandrum')
    property_type = search_params.get('property_type', 'Waterfront')
    price_range = search_params.get('price_range', 'Any')

    predefined_locations = (
        "near latitude 8.3551545319759 and longitude 77.03136608465745, "
        "near latitude 8.414619893463565 and longitude 76.979652, "
        "near latitude 8.438422207850575 and longitude 76.95568054232872, "
        "near latitude 8.612380983078557 and longitude 76.83407053833807"
    )

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=2500
    )

    search = SerperDevTool(api_key=serper_api_key)

    real_estate_agent = Agent(
        llm=llm,
        role="Real Estate Data Specialist",
        goal=(
            f"Find and compile a list of {property_type.lower()} properties for sale "
            f"in {location} within the price range {price_range} (amounts in rupees), specifically in these locations: {predefined_locations}."
        ),
        backstory=(
            "An experienced real estate analyst adept at gathering and verifying property data from multiple sources."
        ),
        allow_delegation=True,
        tools=[search],
        verbose=True
    )

    research_task = Task(
        description=f"""
        Search for {property_type.lower()} properties for sale in {location}.
        Ensure properties have water views, are within the price range {price_range} (amounts in rupees),
        and are located in the following predefined areas: {predefined_locations}.
        Use reputable real estate platforms and provide verified links.
        Format each property as follows:

        'Title: [Name]
        Link: [Verified Link]
        Snippet: [Description] (price in rupees)'
        """,
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
    Enhanced property search with predefined locations.
    """
    try:
        logging.info("Initiating comprehensive property search")

        latitudes = [8.3551545319759, 8.414619893463565, 8.438422207850575, 8.612380983078557]
        longitudes = [77.03136608465745, 76.979652, 76.95568054232872, 76.83407053833807]

        crew = create_real_estate_crew(search_params, latitudes, longitudes)
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
            logging.warning("No properties discovered in search results")
            return None, None

    except Exception as e:
        logging.error(f"Comprehensive search failed: {e}", exc_info=True)
        return None, None

def main():
    st.set_page_config(page_title="Trivandrum Real Estate Intelligence", layout="wide")
    st.title("🏠 Trivandrum Real Estate Intelligence Platform")

    st.sidebar.header("🔍 Property Search Parameters")
    location = st.sidebar.text_input("Location", "Trivandrum")
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
        'price_range': price_range
    }

    if 'df' not in st.session_state:
        st.session_state.df = None

    if st.sidebar.button("🔎 Search Properties"):
        with st.spinner("Conducting comprehensive property search..."):
            df, excel_data = run_property_search(search_params)

            if df is not None and not df.empty:
                st.session_state.df = df
                st.success(f"✅ Found {len(df)} Properties!")
                
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
