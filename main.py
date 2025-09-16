import streamlit as st

st.set_page_config(page_title="Proteomics Data Analysis",
                   page_icon="📊",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None

st.title("Proteomics Data Analysis Platform")

st.markdown("""
## Welcome to the Proteomics Data Analysis Platform

This platform provides comprehensive tools for proteomics data analysis using an integrated example dataset.

### Features:
- Automated data loading and validation
- Advanced data processing (filtering and normalization)
- Statistical analysis and hypothesis testing
- Interactive visualizations and plots
- Publication-ready exports

### Getting Started
The platform automatically loads a comprehensive example proteomics dataset containing:
- 500 proteins with gene identifiers
- Multiple sample groups for comparison
- Quantitative intensity measurements
- Peptide count information

Go to the **Data Configuration** page to set up your sample groups and begin analysis.
""")

st.sidebar.markdown("""
### About
This application is designed for comprehensive proteomics data analysis with interactive statistical tools and visualizations.

#### Dataset
The platform automatically loads a comprehensive example proteomics dataset containing simulated data with multiple cell line sample groups, perfect for demonstrating analysis workflows.

#### User Flow
1. **Data Configuration** → 2. **Data Processing** → 3. **Visualization**

All plots, graphs, and tables are interactive to explore the data. They can be downloaded as interactive HTML files or as publication quality SVG files.

#### Features
- Automated data loading
- Advanced statistical analysis
- Color-blind friendly visualizations
- Publication-ready exports

For questions, please contact: Dr. Nicholas Woods (nicholas.woods@unmc.edu)
""")
