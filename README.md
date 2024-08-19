## Description
Prototype to test whether the hardest things about a forensic accounting and governance AI are feasible.

## Tests
 - Can LLMs extract structured data from SEC filings? [tested]
 - Can LLMs deduce structured data by taking hints from multiple places? [tested]
 - Can LLMs deduce this without being pointed to where the hints are? [todo]

## Setup
1. Copy `.env.example` to `.env`
2. Replace `OpenAI and SEC API keys` in `.env` with your actual API keys
3. Run `python chatAPI.py DKNG` where the last parameter is the ticker of the company you want to test.