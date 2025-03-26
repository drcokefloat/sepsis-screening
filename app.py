import streamlit as st
import pandas as pd
import re
import base64
from io import StringIO, BytesIO
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Sepsis Screening Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'classified_data' not in st.session_state:
    st.session_state.classified_data = None
    
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
    
if 'custom_keywords' not in st.session_state:
    st.session_state.custom_keywords = {"amr": [], "diagnostic": [], "sepsis": []}

if 'errors' not in st.session_state:
    st.session_state.errors = []

# Initialize error logging
def log_error(message, details=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_entry = {
        "timestamp": timestamp,
        "message": message,
        "details": details or "No additional details"
    }
    st.session_state.errors.append(error_entry)
    
# Function to generate download link for error log
def get_error_log_download_link():
    # Convert error log to CSV
    if len(st.session_state.errors) > 0:
        error_df = pd.DataFrame(st.session_state.errors)
        csv = error_df.to_csv(index=False)
        
        # Encode to base64
        csv_b64 = base64.b64encode(csv.encode()).decode()
        
        # Create download link
        href = f'<a href="data:file/csv;base64,{csv_b64}" download="error_log.csv" class="download-link">Download Error Log (CSV)</a>'
        return href
    else:
        return "No errors to download"

# Define keyword groups with their respective terms
amr_keywords = [
    "antibiotic resistance", "antimicrobial resistance", "AMR", 
    "multidrug resistance", "MDR", "drug resistance", 
    "antibacterial resistance", "resistant", "resistances",
    "carbapenem-resistant", "fluoroquinolone-resistant",
    "cephalosporin-resistant", "methicillin-resistant",
    "macrolide-resistant", "ampicillin-resistant",
    "penicillin-resistant", "MRSA"
]

diagnostic_keywords = [
    "diagnos", "test", "assay", "detection", "identify", 
    "identification", "biomarker", "marker", "PCR", "culture",
    "rapid test", "point-of-care", "POC", "diagnostic accuracy",
    "monitoring", "detection", "assessment", "recognition"
]

sepsis_keywords = [
    "sepsis", "septic", "bacteremia", "bloodstream infection", 
    "BSI", "systemic inflammatory response syndrome", "SIRS",
    "septic shock", "severe sepsis", "blood poisoning", 
    "bacterial infection", "endotoxemia", "septicaemia",
    "pyemia", "toxic shock", "sequential organ failure"
]

# Define default keyword dictionary structure
default_keywords = {
    "amr": amr_keywords,
    "diagnostic": diagnostic_keywords,
    "sepsis": sepsis_keywords
}

# Keyword category weights
keyword_weights = {
    "amr": 1.2,      # AMR keywords get 20% higher weight
    "diagnostic": 1.0, # Diagnostic keywords have standard weight
    "sepsis": 1.1     # Sepsis keywords get 10% higher weight
}

# Create dictionary to map keywords to their categories
keyword_categories = {}
for kw in amr_keywords:
    keyword_categories[kw.lower()] = "amr"
for kw in diagnostic_keywords:
    keyword_categories[kw.lower()] = "diagnostic"
for kw in sepsis_keywords:
    keyword_categories[kw.lower()] = "sepsis"

# Function to calculate keyword score with weights
def calculate_score(row, all_keywords):
    try:
        matches = {}
        category_counts = {"amr": 0, "diagnostic": 0, "sepsis": 0}
        
        # Get the columns to search in
        search_fields = []
        for field in ['Title', 'Abstract']:
            if field in row and isinstance(row[field], str):
                search_fields.append(field)
        
        # Fallback: Look for title/abstract in lowercase field names
        if not search_fields:
            for field in row.index:
                if field.lower() in ['title', 'abstract'] and isinstance(row[field], str):
                    search_fields.append(field)
        
        # Last resort: Search in any text columns
        if not search_fields:
            for field in row.index:
                if isinstance(row[field], str) and len(row[field]) > 20:  # Assume longer text fields might be title/abstract
                    search_fields.append(field)
        
        if not search_fields:
            log_error("No searchable fields", f"No Title/Abstract fields found in record. Available fields: {', '.join(row.index)}")
            st.error("No Title or Abstract fields found to search for keywords!")
            return 0, {}, category_counts
            
        # Debug info
        debug_info = f"Searching for keywords in fields: {', '.join(search_fields)}"
        st.info(debug_info)
        
        # Search for keywords in the specified fields
        for category, keywords in all_keywords.items():
            for keyword in keywords:
                if keyword.strip():  # Skip empty keywords
                    score_weight = 1  # Default weight
                    
                    # Apply category-specific weights
                    if category.lower() == "amr":
                        score_weight = 1.5
                    elif category.lower() == "diagnostic":
                        score_weight = 1.0
                    elif category.lower() == "sepsis":
                        score_weight = 1.2
                    
                    # Check for the keyword in each field
                    for field in search_fields:
                        text = str(row[field]).lower()
                        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                        
                        if re.search(pattern, text):
                            if keyword not in matches:
                                matches[keyword] = 0
                            matches[keyword] += score_weight
                            
                            # Increment category count
                            if category.lower() == "amr":
                                category_counts["amr"] += 1
                            elif category.lower() == "diagnostic":
                                category_counts["diagnostic"] += 1
                            elif category.lower() == "sepsis":
                                category_counts["sepsis"] += 1
        
        # Calculate total score
        score = sum(matches.values())
        
        return score, matches, category_counts
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        log_error("Score calculation error", f"Error: {str(e)}\nTrace: {error_trace}")
        st.error(f"Error calculating score: {e}")
        return 0, {}, {"amr": 0, "diagnostic": 0, "sepsis": 0}

# Function to classify records based on score
def classify_record(score, threshold_high, threshold_low):
    if score >= threshold_high:
        return "Include"
    elif score < threshold_low:
        return "Exclude"
    else:
        return "Maybe"

# Function to process the uploaded file
def process_file(file, threshold_high, threshold_low):
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                # Read the uploaded file with specific encoding
                df = pd.read_csv(file, encoding=encoding)
                st.success(f"File successfully read using {encoding} encoding.")
                break  # Exit loop if successful
            except UnicodeDecodeError:
                # If we've tried all encodings and none worked
                if encoding == encodings[-1]:
                    error_msg = f"Failed to read file with any common encoding. Please save your CSV file as UTF-8."
                    st.error(error_msg)
                    log_error("Encoding error", error_msg)
                    return None
                continue  # Try next encoding
            except Exception as e:
                error_msg = f"Error reading file: {e}"
                st.error(error_msg)
                log_error("File reading error", str(e))
                return None
        
        # Display column information for debugging
        st.info(f"Columns found in your data: {', '.join(df.columns.tolist())}")
        
        # Check for required columns (case-insensitive)
        required_fields = ['title', 'abstract']
        df_columns_lower = [col.lower() for col in df.columns]
        
        missing_fields = [field for field in required_fields if field not in df_columns_lower]
        if missing_fields:
            error_msg = f"Required columns missing: {', '.join(missing_fields)}. Your CSV file must have columns named 'Title' and 'Abstract'."
            st.error(error_msg)
            log_error("Missing required columns", f"Missing: {', '.join(missing_fields)}, Available: {', '.join(df.columns)}")
            return None
            
        # Map column names to standard format (case-insensitive)
        column_mapping = {}
        for std_col in ['Title', 'Abstract', 'Authors', 'Year', 'Journal']:
            std_col_lower = std_col.lower()
            for col in df.columns:
                if col.lower() == std_col_lower:
                    column_mapping[col] = std_col
                    break
        
        # Rename columns to standard format if needed
        if column_mapping:
            df = df.rename(columns=column_mapping)
            st.success(f"Columns standardized to: {', '.join([col for col in df.columns if col in ['Title', 'Abstract', 'Authors', 'Year', 'Journal']])}")
        
        # Initialize columns for score and classification
        df['Score'] = 0
        df['AMR_Count'] = 0
        df['Diagnostic_Count'] = 0
        df['Sepsis_Count'] = 0
        df['Matches'] = ""
        df['Classification'] = ""
        
        # Create dictionaries to store all keywords (default + custom)
        all_keywords = {
            "amr": amr_keywords + st.session_state.custom_keywords["amr"],
            "diagnostic": diagnostic_keywords + st.session_state.custom_keywords["diagnostic"], 
            "sepsis": sepsis_keywords + st.session_state.custom_keywords["sepsis"]
        }
        
        # Display preview of data for debugging
        with st.expander("Preview of data (first 3 rows)"):
            st.dataframe(df.head(3))
        
        # Calculate score and classify each record
        for idx, row in df.iterrows():
            score, matches, category_counts = calculate_score(row, all_keywords)
            classification = classify_record(score, threshold_high, threshold_low)
            
            df.at[idx, 'Score'] = round(score, 1)
            df.at[idx, 'AMR_Count'] = category_counts["amr"]
            df.at[idx, 'Diagnostic_Count'] = category_counts["diagnostic"]
            df.at[idx, 'Sepsis_Count'] = category_counts["sepsis"]
            df.at[idx, 'Matches'] = ", ".join(matches.keys())
            df.at[idx, 'Classification'] = classification
        
        # Check if any matches were found
        if df['Score'].sum() == 0:
            warning_msg = "No keyword matches were found in your data. This might indicate that either the data doesn't contain relevant keywords or the column names don't match what the app expects."
            st.warning(warning_msg)
            log_error("No keyword matches", f"Data has correct format but no matches found in {len(df)} records")
        
        return df
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        st.error(f"Error processing file: {e}")
        st.error(f"Detailed error: {error_trace}")
        log_error("Processing error", error_trace)
        return None

# Function to create a download link for the processed data
def get_download_link(df):
    # Prepare columns for export in a logical order
    export_columns = [col for col in df.columns if col not in ['AMR_Count', 'Diagnostic_Count', 'Sepsis_Count', 'Score', 'Matches', 'Classification']]
    export_columns += ['Score', 'AMR_Count', 'Diagnostic_Count', 'Sepsis_Count', 'Matches', 'Classification']
    
    # Create CSV with ordered columns
    csv = df[export_columns].to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    filename = "sepsis_screening_results.csv"
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Screening Results (CSV)</a>'
    return href

# Function to update classification
def update_classification(idx, new_classification):
    if st.session_state.classified_data is not None:
        st.session_state.classified_data.at[idx, 'Classification'] = new_classification

# Function to display and interact with data
def view_data(df, tab_name):
    if len(df) == 0:
        st.write("No records in this category.")
        return
    
    # Display the records in an interactive table
    for idx, row in df.iterrows():
        with st.expander(f"{row.get('Title', f'Record {idx}')} (Score: {row['Score']})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if 'Abstract' in row and pd.notna(row['Abstract']):
                    st.markdown("**Abstract:**")
                    st.write(row['Abstract'])
                
                if 'Authors' in row and pd.notna(row['Authors']):
                    st.markdown("**Authors:**")
                    st.write(row['Authors'])
                
                # Display additional metadata if available
                if 'Year' in row and pd.notna(row['Year']):
                    st.markdown("**Year:**")
                    st.write(row['Year'])
                
                if 'Journal' in row and pd.notna(row['Journal']):
                    st.markdown("**Journal:**")
                    st.write(row['Journal'])
                
                if 'Matches' in row and pd.notna(row['Matches']):
                    st.markdown("**Keyword Matches:**")
                    st.write(row['Matches'])
                    
                # Display category counts
                st.markdown("**Keyword Category Matches:**")
                st.write(f"AMR: {row['AMR_Count']} | Diagnostic: {row['Diagnostic_Count']} | Sepsis: {row['Sepsis_Count']}")
            
            with col2:
                st.markdown("**Current Classification:**")
                st.write(row['Classification'])
                
                # Allow manual classification override
                new_classification = st.selectbox(
                    "Update Classification",
                    options=["Include", "Maybe", "Exclude"],
                    index=["Include", "Maybe", "Exclude"].index(row['Classification']),
                    key=f"class_{tab_name}_{idx}"
                )
                
                if new_classification != row['Classification']:
                    if st.button("Save", key=f"save_{tab_name}_{idx}"):
                        update_classification(idx, new_classification)
                        st.success("Classification updated!")
                        st.experimental_rerun()

# Function to check CSV format
def check_csv_format():
    if uploaded_file is None:
        st.error("Please upload a file first.")
        return
    
    try:
        st.info("Testing different encodings and checking file structure...")
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
        
        # Try to read with different encodings
        success = False
        for encoding in encodings:
            try:
                with st.spinner(f"Trying {encoding} encoding..."):
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    st.success(f"✅ Successfully read file with {encoding} encoding!")
                    log_error("File check", f"Successfully read file '{uploaded_file.name}' using {encoding} encoding")
                    success = True
                    break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                log_error("CSV check error", f"Error with {encoding} encoding: {str(e)}")
                continue
                
        if not success:
            st.error("❌ Could not read the file with any common encoding. Please convert your file to UTF-8 encoding.")
            log_error("CSV encoding error", f"Failed to open file '{uploaded_file.name}' with any encoding")
            return
            
        # Check required columns
        column_info = f"Found columns: {', '.join(df.columns.tolist())}"
        st.info(column_info)
        
        # Check for required columns (case-insensitive)
        df_columns_lower = [col.lower() for col in df.columns]
        if 'title' in df_columns_lower and 'abstract' in df_columns_lower:
            st.success("✅ Required columns 'Title' and 'Abstract' found!")
        else:
            missing = []
            if 'title' not in df_columns_lower:
                missing.append('Title')
            if 'abstract' not in df_columns_lower:
                missing.append('Abstract')
            error_msg = f"❌ Missing required columns: {', '.join(missing)}"
            st.error(error_msg)
            log_error("Missing columns", f"File is missing required columns: {', '.join(missing)}")
            return
            
        # Check for text data
        text_fields = []
        total_rows = len(df)
        empty_titles = 0
        empty_abstracts = 0
        
        for col in df.columns:
            if col.lower() == 'title':
                empty_titles = df[col].isna().sum() + len(df[df[col].astype(str).str.strip() == ''])
                text_fields.append(col)
            elif col.lower() == 'abstract':
                empty_abstracts = df[col].isna().sum() + len(df[df[col].astype(str).str.strip() == ''])
                text_fields.append(col)
        
        if empty_titles > 0:
            warning_msg = f"⚠️ Found {empty_titles} empty or missing values in the Title column ({round(empty_titles/total_rows*100, 1)}% of data)"
            st.warning(warning_msg)
            log_error("Empty data", warning_msg)
        
        if empty_abstracts > 0:
            warning_msg = f"⚠️ Found {empty_abstracts} empty or missing values in the Abstract column ({round(empty_abstracts/total_rows*100, 1)}% of data)"
            st.warning(warning_msg)
            log_error("Empty data", warning_msg)
            
        # Preview the data
        with st.expander("Preview data"):
            st.dataframe(df.head(5))
            
        # Show success if all checks passed
        if empty_titles == 0 and empty_abstracts == 0:
            st.success("✅ CSV format looks good! All checks passed.")
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        st.error(f"Error checking CSV format: {str(e)}")
        log_error("CSV check error", f"Exception: {str(e)}\nTrace: {error_trace}")
        return

# Main app layout
st.title("Sepsis Screening Tool")
st.write("Upload a CSV file containing research records to screen for sepsis and antimicrobial resistance diagnostics.")

# Add tabs for main sections
tab1, tab2, tab3 = st.tabs(["Screening Tool", "Methods Documentation", "Help"])

with tab1:
    # Sidebar for keyword management and file upload
    with st.sidebar:
        st.header("Data Upload")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        # Threshold controls
        st.header("Classification Settings")
        threshold_high = st.slider("Include Threshold", min_value=1, max_value=10, value=3, 
                             help="Records with scores above this threshold will be classified as 'Include'")
        threshold_low = st.slider("Maybe Threshold", min_value=0, max_value=threshold_high, value=1,
                            help="Records with scores above this threshold but below Include will be classified as 'Maybe'")
        
        # Custom keywords section
        st.header("Keyword Management")
        
        with st.expander("Add Custom Keywords"):
            # AMR keywords
            new_amr_keyword = st.text_input("New AMR keyword:", key="new_amr")
            if st.button("Add AMR Keyword"):
                if new_amr_keyword and new_amr_keyword not in amr_keywords and new_amr_keyword not in st.session_state.custom_keywords["amr"]:
                    st.session_state.custom_keywords["amr"].append(new_amr_keyword)
                    st.success(f"Added '{new_amr_keyword}' to AMR keywords!")
                    st.experimental_rerun()
            
            # Diagnostic keywords
            new_diag_keyword = st.text_input("New Diagnostic keyword:", key="new_diag")
            if st.button("Add Diagnostic Keyword"):
                if new_diag_keyword and new_diag_keyword not in diagnostic_keywords and new_diag_keyword not in st.session_state.custom_keywords["diagnostic"]:
                    st.session_state.custom_keywords["diagnostic"].append(new_diag_keyword)
                    st.success(f"Added '{new_diag_keyword}' to Diagnostic keywords!")
                    st.experimental_rerun()
            
            # Sepsis keywords
            new_sepsis_keyword = st.text_input("New Sepsis keyword:", key="new_sepsis")
            if st.button("Add Sepsis Keyword"):
                if new_sepsis_keyword and new_sepsis_keyword not in sepsis_keywords and new_sepsis_keyword not in st.session_state.custom_keywords["sepsis"]:
                    st.session_state.custom_keywords["sepsis"].append(new_sepsis_keyword)
                    st.success(f"Added '{new_sepsis_keyword}' to Sepsis keywords!")
                    st.experimental_rerun()
        
        # Display the current keywords
        with st.expander("View All Keywords"):
            st.subheader("AMR Keywords")
            st.markdown("**Default keywords:**")
            st.write(", ".join(amr_keywords))
            if st.session_state.custom_keywords["amr"]:
                st.markdown("**Custom keywords:**")
                st.write(", ".join(st.session_state.custom_keywords["amr"]))
            
            st.subheader("Diagnostic Keywords")
            st.markdown("**Default keywords:**")
            st.write(", ".join(diagnostic_keywords))
            if st.session_state.custom_keywords["diagnostic"]:
                st.markdown("**Custom keywords:**")
                st.write(", ".join(st.session_state.custom_keywords["diagnostic"]))
            
            st.subheader("Sepsis Keywords")
            st.markdown("**Default keywords:**")
            st.write(", ".join(sepsis_keywords))
            if st.session_state.custom_keywords["sepsis"]:
                st.markdown("**Custom keywords:**")
                st.write(", ".join(st.session_state.custom_keywords["sepsis"]))
        
        # Process the file when uploaded
        if uploaded_file is not None:
            if st.session_state.last_uploaded_file != uploaded_file.name or st.session_state.classified_data is None:
                st.session_state.last_uploaded_file = uploaded_file.name
                with st.spinner("Processing data..."):
                    st.session_state.classified_data = process_file(
                        uploaded_file, threshold_high, threshold_low
                    )
            # Recalculate if custom keywords are added or thresholds changed
            elif any(len(st.session_state.custom_keywords[cat]) > 0 for cat in st.session_state.custom_keywords):
                with st.spinner("Updating classifications with custom keywords..."):
                    # Loop through each record and recalculate score with custom keywords
                    df = st.session_state.classified_data.copy()
                    
                    for idx, row in df.iterrows():
                        score, matches, category_counts = calculate_score(row, all_keywords)
                        classification = classify_record(score, threshold_high, threshold_low)
                        
                        df.at[idx, 'Score'] = round(score, 1)
                        df.at[idx, 'AMR_Count'] = category_counts["amr"]
                        df.at[idx, 'Diagnostic_Count'] = category_counts["diagnostic"]
                        df.at[idx, 'Sepsis_Count'] = category_counts["sepsis"]
                        df.at[idx, 'Matches'] = ", ".join(matches.keys())
                        df.at[idx, 'Classification'] = classification
                    
                    st.session_state.classified_data = df
        
        # Display the results
        if st.session_state.classified_data is not None:
            df = st.session_state.classified_data
            
            # Summary statistics
            st.header("Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Include", len(df[df['Classification'] == 'Include']))
            with col3:
                st.metric("Maybe", len(df[df['Classification'] == 'Maybe']))
            with col4:
                st.metric("Exclude", len(df[df['Classification'] == 'Exclude']))
            
            # Download button
            st.markdown(get_download_link(df), unsafe_allow_html=True)
            
            # Display tabs for different classifications
            st.header("Screening Results")
            result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs(["All Records", "Include", "Maybe", "Exclude"])
            
            with result_tab1:
                view_data(df, "all")
            
            with result_tab2:
                view_data(df[df['Classification'] == 'Include'], "include")
            
            with result_tab3:
                view_data(df[df['Classification'] == 'Maybe'], "maybe")
            
            with result_tab4:
                view_data(df[df['Classification'] == 'Exclude'], "exclude")

with tab2:
    st.header("Methods Documentation")
    
    st.markdown("""
    ## Overview
    
    This tool uses a keyword-based approach to screen research records for relevance to diagnostics in 
    antimicrobial resistance (AMR) in sepsis. The process involves several key steps:
    
    1. **Data Loading**: Reading the uploaded CSV file
    2. **Keyword Matching**: Searching for specific keywords in the title and abstract
    3. **Scoring**: Calculating a relevance score based on weighted keyword matches
    4. **Classification**: Categorizing records based on thresholds
    5. **Manual Review**: Allowing user intervention to refine classifications
    
    Let's explore each step in detail.
    """)
    
    st.subheader("1. Defined Keywords")
    
    st.markdown("""
    Keywords are organized into three categories relevant to our screening criteria:
    
    **AMR Keywords** - Terms related to antimicrobial resistance:
    ```python
    amr_keywords = [
        "antibiotic resistance", "antimicrobial resistance", "AMR", 
        "multidrug resistance", "MDR", "drug resistance", 
        "antibacterial resistance", "resistant", "resistances",
        "carbapenem-resistant", "fluoroquinolone-resistant",
        "cephalosporin-resistant", "methicillin-resistant",
        "macrolide-resistant", "ampicillin-resistant",
        "penicillin-resistant", "MRSA"
    ]
    ```
    
    **Diagnostic Keywords** - Terms related to diagnostic methods:
    ```python
    diagnostic_keywords = [
        "diagnos", "test", "assay", "detection", "identify", 
        "identification", "biomarker", "marker", "PCR", "culture",
        "rapid test", "point-of-care", "POC", "diagnostic accuracy",
        "monitoring", "detection", "assessment", "recognition"
    ]
    ```
    
    **Sepsis Keywords** - Terms related to sepsis and bloodstream infections:
    ```python
    sepsis_keywords = [
        "sepsis", "septic", "bacteremia", "bloodstream infection", 
        "BSI", "systemic inflammatory response syndrome", "SIRS",
        "septic shock", "severe sepsis", "blood poisoning", 
        "bacterial infection", "endotoxemia", "septicaemia",
        "pyemia", "toxic shock", "sequential organ failure"
    ]
    ```
    
    All these keywords are combined into a single list for the screening process, but they retain their category 
    information for weighted scoring.
    """)
    
    st.subheader("2. Weighted Scoring System")
    
    st.markdown("""
    Each keyword category is assigned a different weight to reflect its relative importance:
    
    ```python
    # Keyword category weights
    keyword_weights = {
        "amr": 1.2,      # AMR keywords get 20% higher weight
        "diagnostic": 1.0, # Diagnostic keywords have standard weight
        "sepsis": 1.1     # Sepsis keywords get 10% higher weight
    }
    ```
    
    This means that:
    - AMR-related keywords contribute 20% more to the overall score
    - Sepsis-related keywords contribute 10% more to the overall score
    - Diagnostic-related keywords have a standard weight of 1.0
    
    This weighting system ensures that records specifically addressing antimicrobial resistance in sepsis receive higher scores,
    aligning with the primary focus of this screening tool.
    """)
    
    st.subheader("3. Keyword Matching Algorithm")
    
    st.markdown("""
    The core of the screening process is the keyword matching algorithm. For each record, the algorithm:
    
    1. Searches for each keyword in the title and abstract fields
    2. Uses case-insensitive matching to catch all variations
    3. For longer keywords (>5 characters), uses word boundary matching to find complete words
    4. For shorter keywords, uses substring matching to catch variations
    5. Applies the appropriate weight based on the keyword's category
    6. Keeps track of which keywords matched and in which categories
    
    This logic is implemented in the `calculate_score` function:
    
    ```python
    def calculate_score(row, keywords, fields=['Title', 'Abstract']):
        score = 0
        matches = {}
        category_counts = {"amr": 0, "diagnostic": 0, "sepsis": 0}
        
        # Check which fields are available
        available_fields = [field for field in fields if field in row and pd.notna(row[field])]
        
        if not available_fields:
            # Try with lowercase field names
            lowercase_fields = [field.lower() for field in fields]
            row_lowercase_keys = {k.lower(): k for k in row.keys()}
            available_fields = [row_lowercase_keys.get(field) for field in lowercase_fields if field in row_lowercase_keys and pd.notna(row[row_lowercase_keys.get(field)])]
        
        if not available_fields:
            # Still no fields found, try any text columns as fallback
            text_columns = [col for col in row.keys() if isinstance(row[col], str) and len(str(row[col])) > 20]
            available_fields = text_columns
        
        # Debug info for available fields
        if hasattr(st, 'session_state') and 'debug_info' not in st.session_state:
            st.session_state.debug_info = True
            if available_fields:
                print(f"Searching in fields: {available_fields}")
            else:
                print("Warning: No suitable text fields found for keyword matching")
        
        for field in available_fields:
            text = str(row[field]).lower()
            for keyword in keywords:
                keyword_lower = keyword.lower()
                category = keyword_categories.get(keyword_lower, "other")
                
                # Check for exact match (with word boundaries) or partial match for shorter terms
                if len(keyword_lower) > 5:
                    pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                    if re.search(pattern, text):
                        weight = keyword_weights.get(category, 1.0)
                        score += weight
                        category_counts[category] += 1
                        if keyword not in matches:
                            matches[keyword] = 1
                        else:
                            matches[keyword] += 1
                else:
                    if keyword_lower in text:
                        weight = keyword_weights.get(category, 1.0)
                        score += weight
                        category_counts[category] += 1
                        if keyword not in matches:
                            matches[keyword] = 1
                        else:
                            matches[keyword] += 1
        
        return score, matches, category_counts
    ```
    
    **Technical Details:**
    - Regular expressions (`re` module) are used for pattern matching
    - Word boundary matching (`\\b`) ensures we find whole words, not parts of words
    - `re.escape()` is used to escape any special regex characters in keywords
    - The algorithm tracks both the overall score and the count of matches in each category
    """)
    
    st.subheader("4. Record Classification")
    
    st.markdown("""
    Once a weighted score is calculated for each record, it's classified using configurable thresholds:
    
    ```python
    def classify_record(score, threshold_high, threshold_low):
        if score >= threshold_high:
            return "Include"
        elif score < threshold_low:
            return "Exclude"
        else:
            return "Maybe"
    ```
    
    - Records with scores at or above the high threshold are classified as "Include"
    - Records with scores below the low threshold are classified as "Exclude"
    - Records with scores in between are classified as "Maybe" (requiring manual review)
    
    These thresholds can be adjusted in the sidebar to tune the sensitivity of the screening process.
    """)
    
    st.subheader("5. Technical Implementation Considerations")
    
    st.markdown("""
    **Session State Management**
    
    The app uses Streamlit's session state to persist data between reruns:
    
    ```python
    if 'classified_data' not in st.session_state:
        st.session_state.classified_data = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'threshold_high' not in st.session_state:
        st.session_state.threshold_high = 3
    if 'threshold_low' not in st.session_state:
        st.session_state.threshold_low = 1
    ```
    
    This ensures that:
    - Uploaded data persists when users adjust thresholds
    - Manual classifications are preserved during app interaction
    - Threshold settings are remembered between sessions
    
    **Limitations and Considerations**
    
    This keyword-based approach has some inherent limitations:
    
    1. It treats all keyword matches equally within a category, without considering context or semantics
    2. It doesn't account for negation (e.g., "no evidence of sepsis")
    3. It may miss synonyms or related terms not explicitly included in the keyword lists
    4. It requires manual review of borderline cases ("Maybe" classification)
    
    The manual review feature helps address these limitations by allowing human judgment to refine the automated classifications.
    """)
    
    st.subheader("6. Data Privacy")
    
    st.markdown("""
    **Important privacy note**: All processing happens locally in your browser and on your device. The uploaded data is not sent to any external servers, and no data is stored permanently. When you close the app, all uploaded data is cleared.
    """)

with tab3:
    st.header("Help & Instructions")
    st.markdown("""
    ## How to Use This Tool

    1. **Upload a CSV file** containing your search records. The file should have columns like "Title", "Abstract", and optionally "Authors".
    2. **Adjust the thresholds** in the sidebar to control how records are classified:
       - Records with scores at or above the "Include Threshold" are classified as "Include"
       - Records with scores below the "Exclude Threshold" are classified as "Exclude"
       - Records with scores in between are classified as "Maybe"
    3. **Review the results** in the tabs, which are organized by classification.
    4. **Manually update classifications** if needed by expanding a record and using the dropdown menu.
    5. **Download the results** using the download link provided.

    ## About the Weighted Scoring System

    Each record receives a score based on keyword matches found in the title and abstract.
    Keywords are organized into three categories, each with different weights:
    
    - **AMR Keywords** (Weight: 1.2): Terms related to antimicrobial resistance
    - **Diagnostic Keywords** (Weight: 1.0): Terms related to diagnostic methods and tests
    - **Sepsis Keywords** (Weight: 1.1): Terms related to sepsis and bloodstream infections
    
    The higher weights for AMR and sepsis keywords ensure that records specifically addressing antimicrobial resistance in sepsis receive higher scores, aligning with the screening priorities outlined in the protocol.
    """)

# Troubleshooting section at the bottom of the app
with st.expander("📋 Troubleshooting & Error Log"):
    st.markdown("""
    ### Troubleshooting Tips
    If you're having issues with your data file:
    - Make sure your file is in CSV format (.csv)
    - Check that your file uses UTF-8 encoding
    - Verify that your file has columns named 'Title' and 'Abstract'
    - Preview your data to ensure text is properly formatted
    """)
    
    # Add the CSV check button
    if st.button("🔍 Check CSV File Format"):
        if uploaded_file is not None:
            check_csv_format()
        else:
            st.error("Please upload a file first to check its format.")
    
    # Display error log and download button
    st.markdown("### Error Log")
    if len(st.session_state.errors) == 0:
        st.info("No errors logged yet.")
    else:
        st.warning(f"{len(st.session_state.errors)} errors logged. View details below or download the log file.")
        
        # Create a pandas DataFrame for better display
        error_df = pd.DataFrame(st.session_state.errors)
        st.dataframe(error_df, hide_index=True)
        
        # Add download button
        st.markdown(get_error_log_download_link(), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Sepsis Screening Tool | Created for antimicrobial resistance diagnostics in sepsis research") 