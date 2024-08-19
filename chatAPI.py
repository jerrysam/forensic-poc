import os
import sys
import logging
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from sec_api import QueryApi, ExtractorApi

# Load configuration
load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "100"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_openai_client() -> OpenAI:
    """Set up and return an OpenAI client."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI()


def get_sec_filing_url(ticker: str) -> str:
    """Get the most recent 10-K filing URL for the given ticker."""

    url_file_path = f"database/{ticker}_URL.txt"
    if os.path.exists(url_file_path):
        with open(url_file_path, 'r') as file:
            return file.read().strip()
    
    load_dotenv()
    api_key = os.getenv("SEC_API_KEY")
    queryApi = QueryApi(api_key)
    query = {
        "query": f"ticker:{ticker} AND formType:\"10-K\"",
        "from": "0",
        "size": "1",
        "sort": [{ "filedAt": { "order": "desc" }}]
    }
    filings = queryApi.get_filings(query)
    if filings['filings']:
        result = filings['filings'][0]['linkToFilingDetails']
        with open(url_file_path, "w") as file:
            file.write(result)
        logging.info(f"Filing URL written to {url_file_path}")
        return result
    else:
        raise ValueError(f"No 10-K filings found for {ticker}")

def extract_section_text(filing_url: str, ticker: str) -> str:
    """Extract the relevant section text from the 10-K filing."""
    url_file_path = f"database/{ticker}_text_8.txt"
    if os.path.exists(url_file_path):
        with open(url_file_path, 'r') as file:
            return file.read().strip()

    load_dotenv()
    api_key = os.getenv("SEC_API_KEY")

    extractorApi = ExtractorApi(api_key)
    section_text = extractorApi.get_section(filing_url, "8", "text")
    
    section_title = "Summary of Significant Accounting Policies and Practices"
    result = section_title + section_text.split(section_title, 1)[-1].strip()

    with open(url_file_path, "w") as file:
        file.write(result)
        logging.info(f"Filing URL written to {url_file_path}")

    return result

def prepare_messages(prompt: str, SEC_10k_plaintext: str) -> List[Dict[str, str]]:
    """Prepare the messages for the OpenAI API."""
    return [{"role": "user", "content": f"{prompt}\n\n{SEC_10k_plaintext}"}]


def send_request(client: OpenAI, messages: List[Dict[str, str]], ticker: str) -> str:
    """Send a request to the OpenAI API and return the response."""
    url_file_path = f"database/{ticker}_LLM_response.txt"
    if os.path.exists(url_file_path):
        with open(url_file_path, 'r') as file:
            return file.read().strip()
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS
        )
        result = response.choices[0].message.content
        with open(url_file_path, "w") as file:
            file.write(result)
        logging.info(f"Filing URL written to {url_file_path}")
        return result
    except Exception as e:
        logging.error(f"An error occurred while sending the request: {e}")
        return ""


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    try:

        # Get SEC filing URL
        filing_url = get_sec_filing_url(ticker)
        logging.info(f"Most recent 10-K URL: {filing_url}")

        # Extract relevant section text
        SEC_10k_plaintext = extract_section_text(filing_url, ticker)
        logging.info(f"Extracted {len(SEC_10k_plaintext)} characters")

        prompt = '''
            Based on the attached text from Electronic Arts 10k for year end 2024:
            Question 1: Does the Revenue Recognition section say say that revenue is declared after tax is removed? It might contain phrases like:
            - Net of revenue based taxes
            - Net of product taxes
            - Revenue including taxes
            Question 2: Does the Cost of Revenue section say that the cost of revenue includes taxes?

            Please respond with "yes" or "no" for each question. If the answers are "no, yes", then the last line of your response must say "Yes, there are signs of Aggressive Revenue Recognition". In all other cases, the last line must say "No, there are no signs of Aggressive Revenue Recognition".

            Here are some examples of correct responses:
            "Question 1: no, Question 2: no
            No, there are no signs of Aggressive Revenue Recognition"
            
            "Question 1: yes, Question 2: no
            No, there are no signs of Aggressive Revenue Recognition"
            
            "Question 1: no, Question 2: yes
            Yes, there are signs of Aggressive Revenue Recognition"
            
            "Question 1: yes, Question 2: yes
            Yes, there are signs of Aggressive Revenue Recognition"

            ====
    
            '''
        
        # Set up OpenAI client
        client = setup_openai_client()
        messages = prepare_messages(prompt, SEC_10k_plaintext)
        response = send_request(client, messages, ticker)
        logging.info(response)

    except Exception as e:
        logging.error(f"An error occurred while sending the request: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()