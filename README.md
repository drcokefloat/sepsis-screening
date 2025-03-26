# Sepsis Screening Tool

A Streamlit application for automatically screening research records for diagnostics in antimicrobial resistance in sepsis using a keyword-based approach.

## Features

- Upload CSV files containing search results with columns such as "Title", "Abstract", and "Authors"
- Automatically screen records using keywords from AMR, diagnostic, and sepsis categories
- Calculate scores based on keyword matches in both title and abstract fields
- Classify records as "Include", "Exclude", or "Maybe" based on configurable thresholds
- Manually review and override classifications
- Export final screening results to CSV

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd sepsis_screening
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload your CSV file containing research records (you can use the provided `clean_sample_data.csv` for testing)

4. Adjust the threshold settings in the sidebar as needed

5. Review and manually classify records if necessary

6. Download the results using the provided link

## CSV Encoding Note

If you encounter encoding errors when uploading CSV files, try the following:

1. Use the provided `clean_sample_data.csv` which is guaranteed to be UTF-8 encoded
2. For your own data files:
   - Open your CSV in a text editor like Notepad++ or VS Code
   - Save the file with UTF-8 encoding
   - Alternatively, the app now attempts to handle multiple encodings automatically

## Sample Data

A sample dataset (`clean_sample_data.csv`) is included in this repository for testing purposes. It contains 12 mock research records related to sepsis, antimicrobial resistance, and diagnostic methods. Each record includes title, abstract, authors, publication year, and journal information.

## Input Data Format

The application expects a CSV file with the following columns:
- `Title`: The title of the research paper/record
- `Abstract`: The abstract text of the research paper/record
- Optional columns like `Authors`, `Year`, and `Journal` will also be displayed if present

## Keyword Categories

The screening is based on three categories of keywords:

1. **AMR Keywords**: Terms related to antimicrobial resistance
2. **Diagnostic Keywords**: Terms related to diagnostic methods and tests
3. **Sepsis Keywords**: Terms related to sepsis and bloodstream infections

You can view the specific keywords in each category by expanding the respective sections in the sidebar.

## Weighted Scoring System

Each keyword category has a different weight in the scoring system:
- AMR Keywords: 1.2 (20% higher weight)
- Sepsis Keywords: 1.1 (10% higher weight)
- Diagnostic Keywords: 1.0 (standard weight)

This weighting prioritizes records focusing on antimicrobial resistance in sepsis.

## License

[MIT License](LICENSE) 