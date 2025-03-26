import streamlit as st
import pandas as pd
import re
import base64
from io import StringIO, BytesIO

# Set page config
st.set_page_config(
    page_title="Sepsis Screening Tool",
    page_icon="🔍",
    layout="wide",
)

# Initialize session state for storing manual classifications
if 'classified_data' not in st.session_state:
    st.session_state.classified_data = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'threshold_high' not in st.session_state:
    st.session_state.threshold_high = 3
if 'threshold_low' not in st.session_state:
    st.session_state.threshold_low = 1

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

# Combine all keywords
all_keywords = amr_keywords + diagnostic_keywords + sepsis_keywords

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
def calculate_score(row, keywords, fields=['Title', 'Abstract']):
    score = 0
    matches = {}
    category_counts = {"amr": 0, "diagnostic": 0, "sepsis": 0}
    
    # Check which fields are available
    available_fields = [field for field in fields if field in row and pd.notna(row[field])]
    
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
        # Read the uploaded file
        df = pd.read_csv(file)
        
        # Initialize columns for score and classification
        df['Score'] = 0
        df['AMR_Count'] = 0
        df['Diagnostic_Count'] = 0
        df['Sepsis_Count'] = 0
        df['Matches'] = ""
        df['Classification'] = ""
        
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
        
        return df
    except Exception as e:
        st.error(f"Error processing file: {e}")
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

# Main app layout
st.title("Sepsis Screening Tool")
st.write("Upload a CSV file containing research records to screen for sepsis and antimicrobial resistance diagnostics.")

# Add tabs for main sections
tab1, tab2, tab3 = st.tabs(["Screening Tool", "Methods Documentation", "Help"])

with tab1:
    # Sidebar for settings
    with st.sidebar:
        st.header("Screening Settings")
        
        threshold_high = st.slider(
            "Include Threshold", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.threshold_high,
            help="Records with a score at or above this threshold will be classified as 'Include'"
        )
        
        threshold_low = st.slider(
            "Exclude Threshold", 
            min_value=0, 
            max_value=threshold_high-1, 
            value=min(st.session_state.threshold_low, threshold_high-1),
            help="Records with scores below this threshold will be classified as 'Exclude'"
        )
        
        st.session_state.threshold_high = threshold_high
        st.session_state.threshold_low = threshold_low
        
        st.subheader("Keyword Weights")
        st.info("""
        - AMR Keywords: 1.2 (20% higher weight)
        - Sepsis Keywords: 1.1 (10% higher weight)
        - Diagnostic Keywords: 1.0 (standard weight)
        """)
        
        st.subheader("Keyword Groups")
        with st.expander("AMR Keywords"):
            st.write(", ".join(amr_keywords))
        with st.expander("Diagnostic Keywords"):
            st.write(", ".join(diagnostic_keywords))
        with st.expander("Sepsis Keywords"):
            st.write(", ".join(sepsis_keywords))

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    # Process the file when uploaded
    if uploaded_file is not None:
        if st.session_state.uploaded_file != uploaded_file.name or st.session_state.classified_data is None:
            st.session_state.uploaded_file = uploaded_file.name
            with st.spinner("Processing data..."):
                st.session_state.classified_data = process_file(
                    uploaded_file, 
                    st.session_state.threshold_high, 
                    st.session_state.threshold_low
                )
        
        # If classifications were updated due to threshold changes
        else:
            with st.spinner("Updating classifications..."):
                for idx, row in st.session_state.classified_data.iterrows():
                    classification = classify_record(
                        row['Score'], 
                        st.session_state.threshold_high, 
                        st.session_state.threshold_low
                    )
                    st.session_state.classified_data.at[idx, 'Classification'] = classification
        
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
        
        for field in available_fields:
            text = str(row[field]).lower()
            for keyword in keywords:
                keyword_lower = keyword.lower()
                category = keyword_categories.get(keyword_lower, "other")
                
                # Check for exact match (with word boundaries) or partial match for shorter terms
                if len(keyword_lower) > 5:
                    pattern = r'\\b' + re.escape(keyword_lower) + r'\\b'
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

# Footer
st.markdown("---")
st.caption("Sepsis Screening Tool | Created for antimicrobial resistance diagnostics in sepsis research") 