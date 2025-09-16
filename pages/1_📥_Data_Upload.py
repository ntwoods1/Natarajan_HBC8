import streamlit as st
import pandas as pd
import os
from utils.data_processor import DataProcessor

st.set_page_config(page_title="Data Upload", page_icon="📥")

st.header("Data Upload and Validation")

st.sidebar.markdown("""
### Data Upload Guidelines
Upload your data file or use the example dataset. Files should have a "PG.Genes" column with gene names. 

#### Steps:
1. Choose whether to use the example dataset or upload your own file
2. Name your sample groups
3. Select columns for each group
4. Process your data to continue to the next step

#### File Requirements:
- CSV or Excel format
- Must contain a "PG.Genes" column (or similar protein identifier)
- Quantitative columns should be numeric

You can download the example dataset below to see the recommended format.
""")

# Add download example data button in the sidebar
example_data_path = "data/ExampleData.xlsx"
if os.path.exists(example_data_path):
    with open(example_data_path, "rb") as file:
        st.sidebar.download_button(
            label="Download Example Dataset",
            data=file,
            file_name="ProteomicsExampleData.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download the example dataset to see how your data should be formatted"
        )

# Add a checkbox for using example data
use_example_data = st.checkbox("Use example dataset", value=False, 
                              help="Check this box to use the provided example dataset instead of uploading your own file")

if use_example_data:
    st.info("Using example proteomics dataset")
    
    # First check for original user-uploaded example file
    original_example_path = "ExampleData.xlsx"
    generated_example_path = "data/ExampleData.xlsx"
    
    # Try to use the original uploaded file first, then fall back to generated one
    if os.path.exists(original_example_path):
        example_file_path = original_example_path
        st.success("Using original uploaded example data file")
    elif os.path.exists(generated_example_path):
        example_file_path = generated_example_path
        st.success("Using generated example data file")
    else:
        st.error("Example data file not found. Please upload your own data instead.")
        use_example_data = False
        example_file_path = None
    
    # If we have a valid example file path, load it
    if example_file_path:
        df = pd.read_excel(example_file_path)
        st.write(f"Example dataset contains {df.shape[0]} proteins and {df.shape[1]} columns.")
        
        # Option to automatically set up sample groups with example data
        auto_setup = st.checkbox("Automatically set up sample groups", value=False,
                              help="Automatically configure the sample groups based on the structure of the example dataset")
        
        if auto_setup:
            # Extract potential sample columns (looking for numeric columns that aren't labeled as peptide counts)
            sample_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns 
                          if not ('peptide' in col.lower() or 'sequence' in col.lower())]
            
            # Try to organize columns for the example file
            # This tries to find sample sets by examining the numeric columns
            sample_groups = {}
            
            # Look for patterns in column names to group them
            for col in sample_cols:
                # Try to extract sample prefix/identifier
                parts = col.split('_')
                if len(parts) > 1:
                    # Use first part as group identifier
                    group_prefix = parts[0]
                    if group_prefix not in sample_groups:
                        sample_groups[group_prefix] = []
                    sample_groups[group_prefix].append(col)
            
            # If we couldn't identify specific groups, just split the sample columns
            if not sample_groups and len(sample_cols) >= 3:
                # Split into 3 approximately equal groups
                group_size = len(sample_cols) // 3
                sample_groups = {
                    "Cell Line 1": sample_cols[:group_size],
                    "Cell Line 2": sample_cols[group_size:2*group_size],
                    "Cell Line 3": sample_cols[2*group_size:]
                }
            
            # Display the identified sample groups
            if sample_groups:
                st.markdown("#### Pre-configured Sample Groups:")
                for group_name, columns in sample_groups.items():
                    formatted_name = group_name
                    # Try to map to our default names
                    if group_name.lower() in ['control', 'ctrl', 'c']:
                        formatted_name = "Cell Line 1"
                    elif group_name.lower() in ['treatment', 'treat', 'exp', 't']:
                        formatted_name = "Cell Line 2"
                    
                    st.markdown(f"**{formatted_name}:** {', '.join(columns)}")
                
                # Add explanation about selecting columns
                st.info("Please select the appropriate columns for each sample group. Each group should include replicate measurements of the same experimental condition.")
    else:
        use_example_data = False

if not use_example_data:
    uploaded_file = st.file_uploader(
        "Upload your proteomics data file (CSV or Excel)",
        type=['csv', 'xlsx']
    )

if (use_example_data and example_file_path and os.path.exists(example_file_path)) or (not use_example_data and uploaded_file is not None):
    try:
        # If example data is used, df is already loaded above
        if not use_example_data:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

        st.write("Preview of data:")
        st.dataframe(df.head())

        # Check for PG.Genes column
        if 'PG.Genes' not in df.columns:
            st.error("Required column 'PG.Genes' not found in the dataset")
        else:
            # Get numeric columns for quantitative data selection
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

            # Use a form to batch all inputs
            with st.form("sample_group_form"):
                st.subheader("Configure Sample Groups")
                st.markdown("<h4 style='font-size: 18px; color: orangered;'>If re-naming groups, name all groups first, press enter, then select quantitative columns</h4>", unsafe_allow_html=True)
                
                # Sample group configuration
                # For example data, suggest 3 groups for Cell Lines 1/2/3
                if use_example_data and 'auto_setup' in locals() and auto_setup:
                    # Make sure sample_cols is available in this context
                    if 'sample_cols' not in locals():
                        sample_cols = [col for col in numeric_cols 
                                      if not ('peptide' in col.lower() or 'sequence' in col.lower())]
                        
                        # Here we would look for sample patterns, but we'll let the user select them manually
                        
                    num_groups = 3  # Changed from 2 to 3 for the example file
                    st.info("Using 3 sample groups for the example dataset (Cell Line 1, 2, and 3)")
                else:
                    num_groups = st.number_input("Number of sample groups", min_value=1, max_value=500, value=3)

                # Container for group selections
                group_selections = {}

                # Create selection boxes for each group
                for i in range(num_groups):
                    # Pre-configure group names only (not columns) for example data with auto setup
                    if use_example_data and 'auto_setup' in locals() and auto_setup:
                        if i == 0:
                            default_group_name = "Cell Line 1"
                            default_columns = []  # Don't select columns by default
                        elif i == 1:
                            default_group_name = "Cell Line 2"
                            default_columns = []  # Don't select columns by default
                        elif i == 2:
                            default_group_name = "Cell Line 3"
                            default_columns = []  # Don't select columns by default
                        else:
                            default_group_name = f"Group {i+1}"
                            default_columns = []
                    else:
                        default_group_name = f"Group {i+1}"
                        default_columns = []
                    
                    st.subheader(f"Sample Group {i+1}")
                    group_name = st.text_input(f"Group {i+1} Name", value=default_group_name)
                    selected_cols = st.multiselect(
                        f"Select quantitative columns for {group_name}",
                        options=numeric_cols,
                        default=default_columns,
                        key=f"group_{i}"
                    )
                    group_selections[group_name] = selected_cols

                # Submit button for the form
                submitted = st.form_submit_button("Process Data")

                if submitted:
                    # Check if all groups have selections
                    if all(len(cols) > 0 for cols in group_selections.values()):
                        # Store group information in session state
                        st.session_state.group_selections = group_selections
                        st.session_state.protein_col = 'PG.Genes'

                        # Create a subset of the data with selected columns
                        selected_cols = ['PG.Genes'] + [col for group in group_selections.values() for col in group]
                        
                        # Include peptide count columns if they exist
                        peptide_cols = [col for col in df.columns if 'peptide' in col.lower()]
                        for col in peptide_cols:
                            if col not in selected_cols:
                                selected_cols.append(col)
                                
                        processed_df = df[selected_cols].copy()

                        # Store processed data in session state
                        st.session_state.data = processed_df
                        st.session_state.filtered_data = processed_df.copy()  # For processing page
                        st.session_state.original_full_data = df.copy()  # Store the full original data
                        st.session_state.show_download = True
                        
                        # Add a success message with next steps
                        st.success("Data processed successfully! Go to the Data Processing page for filtering and normalization.")
                    else:
                        st.error("Please select at least one column for each group")

            # Show preview and download button outside the form
            if hasattr(st.session_state, 'show_download') and st.session_state.show_download:
                st.success("Data processed successfully!")
                st.write("Preview of processed data:")
                st.dataframe(st.session_state.data.head())

                # Download processed data
                csv = st.session_state.data.to_csv(index=False)
                st.download_button(
                    label="Download processed data",
                    data=csv,
                    file_name="processed_data.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
