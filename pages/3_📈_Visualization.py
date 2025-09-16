import streamlit as st
from utils.visualizations import Visualizer
from utils.statistics import StatisticalAnalysis
import numpy as np
import pandas as pd
import plotly.express as px
from io import BytesIO
import io
import sys
import subprocess
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Check if we need to install packages for PCA
if 'installing_pca' in st.session_state and st.session_state.installing_pca:
    st.info("Installing scikit-learn for PCA analysis...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    st.session_state.installing_pca = False
    st.success("Installation complete! Ready to generate PCA plot.")

st.set_page_config(page_title="Visualization", page_icon="📈")

st.header("Data Visualization")

st.sidebar.markdown("""
### Visualization Options
Generate interactive plots and visualizations of your processed data.

#### Available Visualizations:
- **Volcano Plot**: Compare differential expression between groups
- **PCA Plot**: Visualize sample clustering in reduced dimensional space
- **Intensity Histograms**: View distribution of expression values
- **Protein Rank Plot**: Analyze overall protein abundance
- **Protein Expression Bar Plot**: Compare expression of specific proteins
- **Correlation Plot**: Examine relationships between samples
- **Heatmap**: Visualize expression patterns across samples
- **Custom Protein Heatmap**: Generate heatmaps for specific proteins

All plots can be downloaded as interactive HTML files or as publication-quality SVG files.
""")

if "data" not in st.session_state or st.session_state.data is None:
    st.warning("Please upload and process your data first.")
else:
    # Use the most recent processed data if available, otherwise use the saved data
    if "filtered_data" in st.session_state and st.session_state.filtered_data is not None:
        df = st.session_state.filtered_data

        # Add a note and refresh button
        st.info("Using the most recent processed data. If you've made changes in the Data Processing page, they will be reflected here.")

        if st.button("Use saved data instead"):
            df = st.session_state.data
            st.success("Using saved data now. Refresh the page to see changes.")
            st.rerun()
    else:
        df = st.session_state.data

    plot_type = st.selectbox(
        "Select Plot Type",
        ["Volcano Plot", "PCA Plot", "Intensity Histograms", "Protein Rank Plot", "Protein Expression Bar Plot", "Correlation Plot", "Heatmap", "Custom Protein Heatmap"]
    )

    # Volcano Plot section
    if plot_type == "Volcano Plot":
        st.subheader("Volcano Plot")

        # Get group selections from session state
        group_selections = st.session_state.get('group_selections', {})

        if not group_selections or len(group_selections) < 2:
            st.warning("You need at least two sample groups for a volcano plot. Please define groups in the Data Upload page.")
        else:
            # Let user select two groups to compare
            group_names = list(group_selections.keys())

            col1, col2 = st.columns(2)
            with col1:
                control_group = st.selectbox("Select control group", group_names, index=0)
            with col2:
                treatment_group = st.selectbox("Select treatment group", 
                                                [g for g in group_names if g != control_group], 
                                                index=0 if len(group_names) > 1 else None)

            # Add fold change threshold slider
            col1, col2 = st.columns(2)
            with col1:
                fc_threshold = st.slider(
                    "Log2 Fold Change threshold",
                    min_value=0.0,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                    help="Proteins with absolute Log2FC greater than this value will be considered significant"
                )

            # Add multiple hypothesis correction options
            correction_method = st.selectbox(
                "Multiple testing correction:",
                ["None", "Benjamini-Hochberg (FDR)", "Bonferroni", "Permutation Test (1000 iterations)"],
                help="Method to correct p-values for multiple hypothesis testing"
            )

            # Add p-value threshold slider after correction method is defined
            with col2:
                # Adjust slider title based on correction method
                p_slider_title = "FDR threshold" if correction_method == "Benjamini-Hochberg (FDR)" else "p-value threshold"
                p_threshold = st.slider(
                    p_slider_title,
                    min_value=0.0001,
                    max_value=0.1,
                    value=0.05,
                    step=0.001,
                    format="%.4f",
                    help="Proteins with p-value/FDR less than this threshold will be considered significant (recommended: 0.05 for regular p-values, 0.01 for FDR control)"
                )

            # Add protein labeling feature
            st.subheader("Protein Labeling")
            proteins_to_label = st.text_area(
                "Enter protein names to label (one per line):",
                help="Enter specific protein names you want to highlight on the volcano plot. Leave empty to use automatic labeling for significant proteins."
            )

            # Show warning and permutation settings if permutation test is selected
            if correction_method == "Permutation Test (1000 iterations)":
                st.warning("⚠️ Permutation test can be time-consuming. For large datasets (>1000 proteins), it may take several minutes.")
                n_permutations = st.slider("Number of permutations:", 100, 1000, 1000, 100,
                                          help="More permutations increase accuracy but take longer")

            # Create a container for the volcano plot to persist
            volcano_container = st.container()

            # Data generation and plot
            if st.button("Generate Volcano Plot"):
                with st.spinner("Calculating fold change and p-values..."):
                    # Get columns for each group
                    control_cols = group_selections[control_group]
                    treatment_cols = group_selections[treatment_group]

                    if not control_cols or not treatment_cols:
                        volcano_container.error("Both groups must have samples defined.")
                    else:
                        # Calculate mean for each group
                        control_mean = df[control_cols].mean(axis=1)
                        treatment_mean = df[treatment_cols].mean(axis=1)

                        # Calculate log2 fold change
                        # Check if data is already log2 transformed (from normalization)
                        is_log2_normalized = st.session_state.get('normalization_method', None) == 'Log2'

                        if is_log2_normalized:
                            # Data is already in log2 space, so fold change is simple subtraction
                            log2fc = treatment_mean - control_mean
                        else:
                            # Data is in linear space, apply log2 transformation
                            # Handle zeros and negative values for log calculation
                            epsilon = 1e-10  # Small value to prevent log(0)
                            ratio = (treatment_mean + epsilon) / (control_mean + epsilon)
                            log2fc = np.log2(ratio)

                        # Calculate p-values using t-test
                        from scipy import stats
                        import numpy as np

                        # Create dataframes to store results
                        result_df = pd.DataFrame({
                            'Log2FC': log2fc
                        })

                        # Calculate p-values
                        p_values = []
                        for index, row in df.iterrows():
                            control_values = row[control_cols].values.astype(float)
                            treatment_values = row[treatment_cols].values.astype(float)

                            # Remove NaN values
                            control_values = control_values[~np.isnan(control_values)]
                            treatment_values = treatment_values[~np.isnan(treatment_values)]

                            if len(control_values) > 0 and len(treatment_values) > 0:
                                try:
                                    # Perform t-test
                                    _, p_val = stats.ttest_ind(
                                        treatment_values, 
                                        control_values, 
                                        equal_var=False,
                                        nan_policy='omit'
                                    )
                                    p_values.append(p_val)
                                except:
                                    p_values.append(1.0)  # Default to 1 on error
                            else:
                                p_values.append(1.0)

                        result_df['p_value'] = p_values

                        # Apply multiple hypothesis correction if selected
                        if correction_method != "None":
                            with st.spinner(f"Applying {correction_method} correction..."):
                                if correction_method == "Benjamini-Hochberg (FDR)":
                                    from statsmodels.stats.multitest import fdrcorrection
                                    _, corrected_p = fdrcorrection(result_df['p_value'], alpha=p_threshold)
                                    result_df['corrected_p_value'] = corrected_p
                                    p_value_col = 'corrected_p_value'
                                    correction_name = "FDR"
                                elif correction_method == "Bonferroni":
                                    from statsmodels.stats.multitest import multipletests
                                    _, corrected_p, _, _ = multipletests(result_df['p_value'], alpha=0.05, method='bonferroni')
                                    result_df['corrected_p_value'] = corrected_p
                                    p_value_col = 'corrected_p_value'
                                    correction_name = "Bonferroni"
                                elif correction_method == "Permutation Test (1000 iterations)":
                                    # Perform permutation test
                                    corrected_p = StatisticalAnalysis.permutation_test(
                                        df, control_cols, treatment_cols, 
                                        p_values, n_permutations=n_permutations
                                    )
                                    result_df['corrected_p_value'] = corrected_p
                                    p_value_col = 'corrected_p_value'
                                    correction_name = "Permutation"
                        else:
                            p_value_col = 'p_value'
                            correction_name = "None"

                        # Get protein names if available
                        protein_col = st.session_state.get('protein_col', None)
                        if protein_col and protein_col in df.columns:
                            result_df['Protein'] = df[protein_col]

                            # Process proteins to label
                            user_proteins_to_label = []
                            if proteins_to_label:
                                user_proteins_to_label = [p.strip() for p in proteins_to_label.split('\n') if p.strip()]

                            # Create a column to indicate which proteins should be labeled
                            result_df['label_protein'] = False
                            if user_proteins_to_label:
                                result_df['label_protein'] = result_df['Protein'].isin(user_proteins_to_label)

                        # Calculate overall study power with fixed alpha=0.01 regardless of user p-value setting
                        with st.spinner(f"Calculating statistical power using proteins that pass the log2 fold-change threshold (alpha=0.01 for power calculation)..."):
                            # Call the power calculation method using proteins filtered by fold-change
                            # Note: We use a fixed alpha=0.01 for power calculation regardless of the p-value threshold for significance
                            power_results = StatisticalAnalysis.calculate_study_power(
                                df, 
                                control_cols, 
                                treatment_cols,
                                alpha=0.01,  # Fixed alpha=0.01 for power analysis
                                fc_threshold=fc_threshold,
                                max_proteins=None
                            )

                            # Extract results
                            mean_power = power_results['mean_power']
                            median_power = power_results['median_power']
                            power_by_effect_size = power_results['power_by_effect_size']
                            sample_info = power_results['sample_size_info']

                            # Store power in session state for use in plot title
                            st.session_state.volcano_power = mean_power

                            # Just store power in session state without displaying the text boxes
                            st.session_state.volcano_power = mean_power

                        # Create volcano plot with custom thresholds and power information
                        fig = Visualizer.create_volcano_plot(
                            result_df, 
                            'Log2FC', 
                            p_value_col,
                            labels=result_df['Protein'].tolist() if 'Protein' in result_df.columns else None,
                            fc_threshold=fc_threshold,
                            p_threshold=p_threshold,
                            correction_name=correction_name,
                            power=mean_power,
                            alpha=0.01 #added alpha for plot title
                        )

                        # Always update the figure and data in session state
                        st.session_state.volcano_fig = fig
                        st.session_state.volcano_result_df = result_df
                        st.session_state.volcano_treatment = treatment_group
                        st.session_state.volcano_control = control_group

            # Display plot if it exists in session state
            with volcano_container:
                if 'volcano_fig' in st.session_state:
                    st.plotly_chart(st.session_state.volcano_fig)

            # Add download options in a separate container below the plot
            if 'volcano_fig' in st.session_state and 'volcano_result_df' in st.session_state:
                download_container = st.container()
                with download_container:
                    st.write("### Download Options")
                    col1, col2, col3 = st.columns(3)

                    # Get result dataframe from session state
                    result_df = st.session_state.volcano_result_df
                    fig = st.session_state.volcano_fig


                    # Generate fresh CSV data for this specific comparison
                    csv_key = f"csv_data_{st.session_state.volcano_treatment}_{st.session_state.volcano_control}"
                    st.session_state[csv_key] = result_df.to_csv(index=False)

                    # CSV download
                    with col1:
                        file_name = f"volcano_plot_{st.session_state.volcano_treatment}_vs_{st.session_state.volcano_control}.csv"
                        st.download_button(
                            "Download Results (CSV)",
                            st.session_state[csv_key],
                            file_name=file_name,
                            mime="text/csv",
                            key=f"csv_download_{st.session_state.volcano_treatment}_{st.session_state.volcano_control}"
                        )

                    # Generate fresh HTML data for this specific comparison
                    html_key = f"html_data_{st.session_state.volcano_treatment}_{st.session_state.volcano_control}"
                    buffer = io.StringIO()
                    fig.write_html(buffer)
                    st.session_state[html_key] = buffer.getvalue().encode()

                    # HTML download
                    with col2:
                        file_name = f"volcano_plot_{treatment_group}_vs_{control_group}.html"
                        st.download_button(
                            "Download Interactive Plot (HTML)",
                            st.session_state[html_key],
                            file_name=file_name,
                            mime="text/html",
                            key=f"html_download_{treatment_group}_{control_group}"
                        )

                    # SVG download
                    with col3:
                        try:
                            # Generate fresh SVG data for this specific comparison
                            svg_key = f"svg_data_{st.session_state.volcano_treatment}_{st.session_state.volcano_control}"
                            buffer = io.BytesIO()
                            fig.write_image(buffer, format="svg")
                            st.session_state[svg_key] = buffer.getvalue()

                            file_name = f"volcano_plot_{st.session_state.volcano_treatment}_vs_{st.session_state.volcano_control}.svg"
                            st.download_button(
                                "Download Plot (SVG)",
                                st.session_state[svg_key],
                                file_name=file_name,
                                mime="image/svg+xml",
                                key=f"svg_download_{st.session_state.volcano_treatment}_{st.session_state.volcano_control}"
                            )
                        except Exception as e:
                            st.error(f"SVG export failed: {str(e)}")
                            st.info("Please try refreshing the page after installation of required packages is complete.")

            # Display top DE proteins if data exists
            if 'volcano_result_df' in st.session_state:
                top_proteins_container = st.container()
                with top_proteins_container:
                    st.subheader("Top Differentially Expressed Proteins")

                    # Get result data from session state
                    result_df = st.session_state.volcano_result_df

                    # Add significance column (-log10 p-value)
                    result_df['-log10(p)'] = -np.log10(result_df['p_value'])

                    # Sort by significance and fold change
                    significant_df = result_df.copy()
                    significant_df['abs_log2FC'] = result_df['Log2FC'].abs()
                    significant_df = significant_df.sort_values(by=['-log10(p)', 'abs_log2FC'], ascending=False)

                    if 'Protein' in significant_df.columns:
                        significant_df = significant_df[['Protein', 'Log2FC', 'p_value', '-log10(p)']]

                    st.dataframe(significant_df.head(20))

    elif plot_type == "PCA Plot":
        st.subheader("Principal Component Analysis (PCA)")


        # Get group selections from session state
        group_selections = st.session_state.get('group_selections', {})


        if not group_selections:
            st.warning("You need to define sample groups for PCA. Please define groups in the Data Upload page.")
        else:
            st.write("""
            PCA reduces the dimensionality of the data to visualize similarities and differences between samples.
            Each point represents a sample, and the distance between points represents similarity.
            """)


            # Allow user to select which groups to include
            st.subheader("Select Groups to Include")


            selected_groups = {}
            for group_name, columns in group_selections.items():
                if st.checkbox(f"Include {group_name}", value=True):
                    selected_groups[group_name] = columns


            # Option to show confidence ellipses
            show_ellipses = st.radio(
                "Show confidence ellipses",
                ["No", "Yes"],
                index=0
            ) == "Yes"

            if show_ellipses:
                confidence_level = st.slider(
                    "Confidence Level (%)",
                    min_value=90,
                    max_value=99,
                    value=95,
                    help="Select the confidence level for the ellipses (90-100%)"
                )

            # Create a container for the PCA plot
            pca_container = st.container()

            if st.button("Generate PCA Plot"):
                with st.spinner("Performing PCA analysis..."):
                    try:
                        # Check if we have sklearn installed
                        try:
                            from sklearn.decomposition import PCA
                        except ImportError:
                            st.info("Installing required packages for PCA...")
                            st.session_state.installing_pca = True
                            st.rerun()

                        # Create the PCA plot with selected confidence level
                        confidence_level_decimal = confidence_level / 100 if show_ellipses else None
                        fig = Visualizer.create_pca_plot(df, selected_groups, show_ellipses, confidence_level_decimal)

                        # Store in session state
                        st.session_state.pca_fig = fig
                    except Exception as e:
                        st.error(f"Error generating PCA plot: {str(e)}")


            # Display plot if it exists in session state
            with pca_container:
                if 'pca_fig' in st.session_state:
                    st.plotly_chart(st.session_state.pca_fig)

                    # Add download options
                    st.write("### Download Options")
                    col1, col2 = st.columns(2)


                    # Get figure from session state
                    fig = st.session_state.pca_fig


                    # HTML download
                    with col1:
                        import io
                        buffer = io.StringIO()
                        # Make sure figure is saved with full color information
                        fig.update_layout(
                            coloraxis_showscale=True,
                            colorway=px.colors.qualitative.Plotly  # Ensure colors are defined
                        )
                        fig.write_html(buffer, include_plotlyjs='cdn')
                        html_data = buffer.getvalue().encode()


                        st.download_button(
                            "Download Interactive Plot (HTML)",
                            html_data,
                            file_name="pca_plot.html",
                            mime="text/html",
                            key="html_download_pca"
                        )


                    # SVG download
                    with col2:
                        try:
                            import plotly.io as pio


                            # Set specific configuration to preserve colors
                            # First, ensure we have explicit colors for each trace
                            for i, trace in enumerate(fig.data):
                                if trace.mode == 'markers' and hasattr(trace, 'name'):
                                    # Get the color from plotly's default color sequence if not set
                                    if not hasattr(trace.marker, 'color') or trace.marker.color is None:
                                        color_idx = i % len(px.colors.qualitative.Plotly)
                                        trace.marker.color = px.colors.qualitative.Plotly[color_idx]


                            # Set higher resolution for better quality
                            pio.kaleido.scope.default_scale = 2.0


                            # Use write_image with proper configuration
                            buffer = io.BytesIO()
                            fig.write_image(
                                buffer, 
                                format="svg", 
                                engine="kaleido",
                                width=1000,  # Larger width for better quality
                                height=800   # Larger height for better quality
                            )
                            svg_data = buffer.getvalue()


                            st.download_button(
                                "Download Plot (SVG)",
                                svg_data,
                                file_name="pca_plot.svg",
                                mime="image/svg+xml",
                                key="svg_download_pca"
                            )
                        except Exception as e:
                            st.error(f"SVG export failed: {str(e)}")
                            st.info("Please try installing kaleido with: pip install kaleido")

    elif plot_type == "Intensity Histograms":
        st.subheader("Protein Intensity Distributions")

        # Get group selections from session state
        group_selections = st.session_state.get('group_selections', {})

        if not group_selections:
            st.warning("You need to define sample groups for intensity histograms. Please define groups in the Data Upload page.")
        else:
            st.write("""
            These histograms show the distribution of protein intensities across all samples.
            Each plot represents a sample, and the distribution shows how many proteins are detected at different intensity levels.
            """)

            # Allow user to select which groups to include
            st.subheader("Select Groups to Include")

            selected_groups = {}
            for group_name, columns in group_selections.items():
                if st.checkbox(f"Include {group_name}", value=True):
                    selected_groups[group_name] = columns

            # Create a container for the histogram plots
            histogram_container = st.container()

            if st.button("Generate Intensity Histograms"):
                with st.spinner("Generating intensity histograms..."):
                    try:
                        # Create the intensity histograms
                        fig = Visualizer.create_intensity_histograms(df, selected_groups)

                        # Store in session state
                        st.session_state.intensity_fig = fig
                    except Exception as e:
                        st.error(f"Error generating intensity histograms: {str(e)}")

            # Display plot if it exists in session state
            with histogram_container:
                if 'intensity_fig' in st.session_state:
                    st.plotly_chart(st.session_state.intensity_fig, use_container_width=True)

                    # Add download options
                    st.write("### Download Options")
                    col1, col2 = st.columns(2)

                    # Get figure from session state
                    fig = st.session_state.intensity_fig

                    # HTML download
                    with col1:
                        import io
                        buffer = io.StringIO()
                        fig.write_html(buffer, include_plotlyjs='cdn')
                        html_data = buffer.getvalue().encode()

                        st.download_button(
                            "Download Interactive Plot (HTML)",
                            html_data,
                            file_name="intensity_histograms.html",
                            mime="text/html",
                            key="html_download_intensity"
                        )

                    # SVG download
                    with col2:
                        try:
                            import plotly.io as pio

                            # Set specific configuration to preserve colors
                            # First, ensure we have explicit colors for each trace
                            for i, trace in enumerate(fig.data):
                                if hasattr(trace, 'marker') and not hasattr(trace.marker, 'color'):
                                    color_idx = i % len(px.colors.qualitative.Plotly)
                                    trace.marker.color = px.colors.qualitative.Plotly[color_idx]

                            # Set higher resolution for better quality
                            pio.kaleido.scope.default_scale = 2.0

                            # Use write_image with proper configuration
                            buffer = io.BytesIO()
                            fig.write_image(
                                buffer, 
                                format="svg", 
                                engine="kaleido",
                                width=1200,  # Larger width for better quality
                                height=800   # Larger height for better quality
                            )
                            svg_data = buffer.getvalue()

                            st.download_button(
                                "Download Plot (SVG)",
                                svg_data,
                                file_name="expression_histograms.svg",
                                mime="image/svg+xml",
                                key="svg_download_intensity"
                            )
                        except Exception as e:
                            st.error(f"SVG export failed: {str(e)}")
                            st.info("Please try installing kaleido with: pip install kaleido")

    elif plot_type == "Protein Rank Plot":
        st.subheader("Protein Rank Plot")

        # Option to use unfiltered data
        use_unfiltered = st.checkbox("Use unfiltered data", 
                                    help="When enabled, shows the original unfiltered data instead of filtered data")

        # Get data to plot based on checkbox
        if use_unfiltered:
            data_to_plot = st.session_state.data

            # Only show highlight option if both filtered and unfiltered data exist
            highlight_removed = False
            if "filtered_data" in st.session_state and st.session_state.filtered_data is not None:
                highlight_removed = st.checkbox("Highlight removed proteins", 
                                               help="Highlight proteins that were removed during filtering")
            filtered_data = st.session_state.get("filtered_data", None)
        else:
            data_to_plot = st.session_state.filtered_data
            highlight_removed = False
            filtered_data = None

        if data_to_plot is not None:
            # Get group selections from session state to use for intensity columns
            group_selections = st.session_state.get('group_selections', {})

            # Create a list of all intensity columns from groups
            all_intensity_columns = []
            for group_name, columns in group_selections.items():
                all_intensity_columns.extend(columns)

            # If no groups are defined, use all numeric columns
            if not all_intensity_columns:
                all_intensity_columns = data_to_plot.select_dtypes(include=['number']).columns.tolist()

            # Select columns to use for calculating abundance
            selected_columns = st.multiselect(
                "Select columns for calculating protein abundance",
                options=all_intensity_columns,
                default=[],  # No preselected columns
                help="Select which columns to use for calculating mean protein abundance"
            )

            # Protein highlighting section
            st.subheader("Highlight Proteins")
            proteins_to_highlight_input = st.text_area(
                "Enter protein names to highlight (one per line):",
                help="Enter protein names to highlight on the plot. Leave blank to not highlight any proteins."
            )
            proteins_to_highlight = []
            if proteins_to_highlight_input:
                proteins_to_highlight = [p.strip() for p in proteins_to_highlight_input.split('\n') if p.strip()]


            # Generate the plot
            if selected_columns and st.button("Generate Protein Rank Plot"):
                try:
                    with st.spinner("Generating Protein Rank Plot..."):
                        # Use the visualizer to create the plot
                        fig = Visualizer.create_protein_rank_plot(
                            data_to_plot, 
                            selected_columns,
                            highlight_removed=highlight_removed,
                            filtered_data=filtered_data,
                            proteins_to_highlight=proteins_to_highlight
                        )

                        # Store in session state
                        st.session_state.protein_rank_fig = fig

                except Exception as e:
                    st.error(f"Error generating Protein Rank Plot: {e}")

            # Display plot if it exists in session state
            if 'protein_rank_fig' in st.session_state:
                st.plotly_chart(st.session_state.protein_rank_fig, use_container_width=True)

                # Add download options
                st.write("### Download Options")
                col1, col2 = st.columns(2)

                # HTML download
                with col1:
                    buffer = io.StringIO()
                    # Make sure colors are explicitly defined for download
                    fig = st.session_state.protein_rank_fig
                    fig.update_layout(
                        coloraxis_showscale=True,
                        colorway=px.colors.qualitative.Plotly
                    )
                    # Ensure each trace has explicit colors
                    for i, trace in enumerate(fig.data):
                        if hasattr(trace, 'marker') and not hasattr(trace.marker, 'color'):
                            color_idx = i % len(px.colors.qualitative.Plotly)
                            trace.marker.color = px.colors.qualitative.Plotly[color_idx]

                    fig.write_html(buffer, include_plotlyjs='cdn')
                    html_data = buffer.getvalue().encode()

                    st.download_button(
                        "Download Interactive Plot (HTML)",
                        html_data,
                        file_name="protein_rank_plot.html",
                        mime="text/html",
                        key="html_download_rank"
                    )

                # SVG download
                with col2:
                    try:
                        import plotly.io as pio

                        # Set higher resolution for better quality
                        pio.kaleido.scope.default_scale = 2.0

                        # Ensure colors are explicitly set for all traces
                        fig = st.session_state.protein_rank_fig
                        for i, trace in enumerate(fig.data):
                            if hasattr(trace, 'marker') and not hasattr(trace.marker, 'color'):
                                color_idx = i % len(px.colors.qualitative.Plotly)
                                trace.marker.color = px.colors.qualitative.Plotly[color_idx]

                        # Use write_image with proper configuration
                        buffer = io.BytesIO()
                        fig.write_image(
                            buffer, 
                            format="svg", 
                            engine="kaleido",
                            width=1200,
                            height=800
                        )
                        svg_data = buffer.getvalue()

                        st.download_button(
                            "Download Plot (SVG)",
                            svg_data,
                            file_name="protein_rank_plot.svg",
                            mime="image/svg+xml",
                            key="svg_download_rank"
                        )
                    except Exception as e:
                        st.error(f"SVG export failed: {str(e)}")
                        st.info("Please try installing kaleido with: pip install kaleido")
        else:
            st.warning("No data available for plotting. Please upload and process data first.")

    elif plot_type == "Protein Expression Bar Plot":
        st.subheader("Protein Expression Bar Plot")

        # Add explanation of statistical methods
        st.markdown("""
        ### Statistical Analysis Method

        This visualization compares protein expression between different groups using **Welch's t-test**, 
        which does not assume equal variances between groups. This is the same statistical test used in the volcano plot.

        * **p < 0.05** = * (significant)
        * **p < 0.01** = ** (highly significant)
        * **p < 0.001** = *** (extremely significant)
        * **p ≥ 0.05** = ns (not significant)

        The bar heights represent the mean expression value for each group, and error bars show the standard error of the mean (SEM).
        Individual data points are plotted to show the distribution of values within each group.
        """)

        # Get group selections from session state
        group_selections = st.session_state.get('group_selections', {})

        if not group_selections:
            st.warning("No sample groups defined. Please configure groups in the Data Upload page.")
        else:
            # Option to use unfiltered data
            use_unfiltered = st.checkbox("Use unfiltered data", 
                                        help="When enabled, shows the original unfiltered data instead of filtered data")

            # Get data to plot based on checkbox
            if use_unfiltered:
                data_to_plot = st.session_state.data
            else:
                data_to_plot = st.session_state.filtered_data

            if data_to_plot is not None:
                # Provide option to select proteins
                protein_col = st.session_state.get('protein_col', None)

                # Try to find a protein column if not set in session state
                if not protein_col or protein_col not in data_to_plot.columns:
                    for col in data_to_plot.columns:
                        if 'protein' in col.lower() or 'gene' in col.lower() or 'pg.genes' in col.lower():
                            protein_col = col
                            break

                # If protein column found, allow user to select proteins
                if protein_col and protein_col in data_to_plot.columns:
                    # Create searchable dropdown for protein selection
                    st.write("### Select Proteins to Plot")

                    # Input box for protein names
                    protein_input = st.text_area(
                        "Enter protein names to plot (one per line):",
                        help="Enter specific protein names to visualize on thebar plot."
                    )

                    # Process the input
                    proteins_to_plot = []
                    if protein_input:
                        proteins_to_plot = [p.strip() for p in protein_input.split('\n') if p.strip()]

                    # Validate if proteins exist in the data
                    valid_proteins = []
                    for protein in proteins_to_plot:
                        if protein in data_to_plot[protein_col].values:
                            valid_proteins.append(protein)

                    if not valid_proteins and proteins_to_plot:
                        st.warning(f"None of the specified proteins were found in the data. Please check the protein names.")

                    # Allow users to select which groups to include in the plot
                    st.subheader("Select Groups to Include in Comparison")
                    selected_groups = {}
                    cols = st.columns(min(3, len(group_selections)))
                    for i, group_name in enumerate(group_selections.keys()):
                        col_idx = i % len(cols)
                        with cols[col_idx]:
                            selected_groups[group_name] = st.checkbox(f"{group_name}", value=True, key=f"bar_group_{group_name}")

                    # Check if at least two groups are selected for statistical comparison
                    groups_selected = sum(selected_groups.values())
                    if groups_selected < 2:
                        st.warning("Please select at least two groups for statistical comparison.")

                    # Generate the plot when button is clicked
                    if st.button("Generate Protein Expression Plot") and valid_proteins and groups_selected >= 2:
                        with st.spinner("Generating protein expression plot..."):
                            try:
                                # Create bar plot with Visualizer - now returns dictionary of figures
                                protein_figs, stats_df = Visualizer.create_protein_bar_plot(
                                    data_to_plot, 
                                    valid_proteins, 
                                    group_selections,
                                    selected_groups
                                )

                                # Store in session state
                                st.session_state.protein_bar_figs = protein_figs
                                st.session_state.protein_bar_stats = stats_df
                            except Exception as e:
                                st.error(f"Error generating protein expression plot: {str(e)}")
                                st.info("Please check your data and try again.")

                    # Display plots after generation
                    if 'protein_bar_figs' in st.session_state:
                        try:
                            # Get the protein figures and stats from session state
                            protein_figs = st.session_state.protein_bar_figs
                            stats_df = st.session_state.protein_bar_stats

                            # Save images for future reference
                            st.session_state.protein_bar_images = protein_figs

                            # Display each protein figure separately 
                            for protein_name, (img, svg_data) in protein_figs.items():
                                st.subheader(f"{protein_name}")
                                st.image(img, use_container_width=True)

                                # Add download buttons right after each figure
                                col1, col2 = st.columns(2)

                                with col1:
                                    # Generate fresh PNG buffer for download
                                    buf = io.BytesIO()
                                    img.save(buf, format='PNG')
                                    buf.seek(0)

                                    # Use unique key with protein name for download
                                    st.download_button(
                                        f"Download {protein_name} Plot (PNG)",
                                        buf,
                                        file_name=f"{protein_name}_expression_plot.png",
                                        mime="image/png",
                                        key=f"png_download_{protein_name}"
                                    )

                                with col2:
                                    # Use unique key with protein name for SVG download
                                    st.download_button(
                                        f"Download {protein_name} Plot (SVG)",
                                        svg_data,
                                        file_name=f"{protein_name}_expression_plot.svg",
                                        mime="image/svg+xml",
                                        key=f"svg_download_{protein_name}"
                                    )

                            # Display statistics table
                            if not stats_df.empty:
                                st.write("### Statistical Comparison")
                                st.dataframe(stats_df)

                                # Add CSV download for statistics
                                stats_csv = stats_df.to_csv(index=False)
                                st.download_button(
                                    "Download Statistics (CSV)",
                                    stats_csv,
                                    file_name="protein_expression_stats.csv",
                                    mime="text/csv"
                                )
                        except Exception as e:
                            st.error(f"Error generating protein expression plot: {str(e)}")
                            st.info("Please check your data and try again.")

                    # Display previously generated plots if they exist but no new plots were generated
                    elif 'protein_bar_images' in st.session_state:
                        st.success("Displaying previously generated plots. Click 'Generate Protein Expression Plot' to refresh.")

                        # Display each saved protein figure
                        for protein_name, (img, svg_data) in st.session_state.protein_bar_images.items():
                            st.subheader(f"{protein_name}")
                            st.image(img, use_container_width=True)

                            # Add download buttons for each figure
                            col1, col2 = st.columns(2)

                            with col1:
                                # Generate fresh PNG buffer for download
                                buf = io.BytesIO()
                                img.save(buf, format='PNG')
                                buf.seek(0)

                                st.download_button(
                                    f"Download {protein_name} Plot (PNG)",
                                    buf,
                                    file_name=f"{protein_name}_expression_plot.png",
                                    mime="image/png",
                                    key=f"png_download_saved_{protein_name}"
                                )

                            with col2:
                                st.download_button(
                                    f"Download {protein_name} Plot (SVG)",
                                    svg_data,
                                    file_name=f"{protein_name}_expression_plot.svg",
                                    mime="image/svg+xml",
                                    key=f"svg_download_saved_{protein_name}"
                                )

                        # Display statistics table if it exists
                        if 'protein_bar_stats' in st.session_state and not st.session_state.protein_bar_stats.empty:
                            st.write("### Statistical Comparison")
                            st.dataframe(st.session_state.protein_bar_stats)

                            # Add CSV download for statistics
                            stats_csv = st.session_state.protein_bar_stats.to_csv(index=False)
                            st.download_button(
                                "Download Statistics (CSV)",
                                stats_csv,
                                file_name="protein_expression_stats.csv",
                                mime="text/csv",
                                key="stats_csv_saved"
                            )
                else:
                    st.error("No protein identifier column found in the data. Please make sure your data includes protein IDs.")
            else:
                st.warning("No data available for plotting. Please upload and process data first.")

    elif plot_type == "Correlation Plot":
        st.subheader("Sample Correlation Plot")

        # Get group selections from session state
        group_selections = st.session_state.get('group_selections', {})

        if not group_selections:
            st.warning("You need to define sample groups. Please define groups in the Data Upload page.")
        else:
            # Let user choose correlation mode
            correlation_mode = st.radio(
                "Select correlation analysis mode:",
                ["Within Groups", "Between Groups"],
                help="Choose to view correlations within sample groups or between group averages"
            )

            if correlation_mode == "Within Groups":
                # Allow user to select which groups to include
                st.subheader("Select Groups to Include")
                selected_groups = {}
                for group_name, columns in group_selections.items():
                    if st.checkbox(f"Include {group_name}", value=True, key=f"corr_{group_name}"):
                        selected_groups[group_name] = columns

                if selected_groups and st.button("Generate Correlation Plot"):
                    with st.spinner("Generating correlation plot..."):
                        figs, stats_df = Visualizer.create_correlation_plot(df, selected_groups, mode="within_groups")
                        st.session_state.within_groups_figs = {}
                        st.session_state.within_groups_data = {}
                        st.session_state.correlation_stats = stats_df

                        # Store figures and data in session state
                        for idx, fig in enumerate(figs):
                            group_name = list(selected_groups.keys())[idx]
                            st.session_state.within_groups_figs[group_name] = fig

                            # Store HTML and SVG data
                            buffer = io.StringIO()
                            fig.write_html(buffer, include_plotlyjs='cdn')
                            buffer2 = io.BytesIO()
                            fig.write_image(buffer2, format="svg", width=1000, height=800)

                            st.session_state.within_groups_data[group_name] = {
                                'html': buffer.getvalue().encode(),
                                'svg': buffer2.getvalue()
                            }

                # Display plots from session state
                if 'within_groups_figs' in st.session_state:
                    for group_name, fig in st.session_state.within_groups_figs.items():
                        st.plotly_chart(fig, use_container_width=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "Download Interactive Plot (HTML)",
                                st.session_state.within_groups_data[group_name]['html'],
                                file_name=f"correlation_plot_{group_name}.html",
                                mime="text/html",
                                key=f"html_download_{group_name}"
                            )
                        with col2:
                            st.download_button(
                                "Download Plot (SVG)",
                                st.session_state.within_groups_data[group_name]['svg'],
                                file_name=f"correlation_plot_{group_name}.svg",
                                mime="image/svg+xml",
                                key=f"svg_download_{group_name}"
                            )
                        st.markdown("---")


                # Display correlation statistics table
                if 'correlation_stats' in st.session_state:
                    st.write("### Correlation Statistics")
                    st.dataframe(st.session_state.correlation_stats)


                    # Add CSV download for statistics
                    stats_csv = st.session_state.correlation_stats.to_csv(index=False)
                    st.download_button(
                        "Download Correlation Statistics (CSV)",
                        stats_csv,
                        file_name="correlation_statistics.csv",
                        mime="text/csv",
                        key="correlation_stats_download"
                    )

            else:  # Between Groups
                # Let user select two groups to compare
                col1, col2 = st.columns(2)
                with col1:
                    group1 = st.selectbox("Select first group", options=list(group_selections.keys()), key="corr_group1")
                with col2:
                    group2 = st.selectbox("Select second group", 
                                        options=[g for g in group_selections.keys() if g != group1],
                                        key="corr_group2")

                # Generate button moved to plot generation logic above
                if st.button("Generate Correlation Plot"):
                    with st.spinner("Generating correlation plot..."):
                        fig = Visualizer.create_correlation_plot(
                            df, 
                            group_selections,
                            mode="between_groups",
                            group1=group1,
                            group2=group2
                        )
                        # Store figure and groups in session state with unique keys
                        st.session_state.between_groups_fig = fig
                        st.session_state.between_groups_group1 = group1
                        st.session_state.between_groups_group2 = group2

                        # Store HTML and SVG data with unique keys
                        buffer = io.StringIO()
                        fig.write_html(buffer, include_plotlyjs='cdn')
                        st.session_state.between_groups_html = buffer.getvalue().encode()

                        buffer = io.BytesIO()
                        fig.write_image(buffer, format="svg", width=1000, height=800)
                        st.session_state.between_groups_svg = buffer.getvalue()

                # Display plot if it exists in session state
                if 'between_groups_fig' in st.session_state:
                    st.plotly_chart(st.session_state.between_groups_fig, use_container_width=True)

                    # Add download options centered
                    st.write("### Download Options")
                    _, center_col, _ = st.columns([1,2,1])
                    with center_col:
                        st.download_button(
                            "Download Interactive Plot (HTML)",
                            st.session_state.between_groups_html,
                            file_name=f"correlation_plot_{st.session_state.between_groups_group1}_vs_{st.session_state.between_groups_group2}.html",
                            mime="text/html",
                            key=f"html_download_between_groups"
                        )
                        st.download_button(
                            "Download Plot (SVG)",
                            st.session_state.between_groups_svg,
                            file_name=f"correlation_plot_{st.session_state.between_groups_group1}_vs_{st.session_state.between_groups_group2}.svg",
                            mime="image/svg+xml",
                            key=f"svg_download_between_groups"
                        )

    elif plot_type == "Heatmap":
        st.subheader("Protein Expression Heatmap")

        # Get group selections
        group_selections = st.session_state.get('group_selections', {})

        if not group_selections:
            st.warning("Please define sample groups in the Data Upload page first.")
        else:
            # Create a container for the heatmap to persist
            heatmap_display_container = st.container()

            # Settings area
            settings_container = st.container()
            with settings_container:
                # Number of proteins to display
                n_proteins = st.slider("Number of proteins to display", 10, 200, 50)

                # Clustering options
                col1, col2 = st.columns(2)
                with col1:
                    cluster_rows = st.toggle("Cluster proteins", value=True)
                with col2:
                    cluster_cols = st.toggle("Cluster samples", value=True)

                # Z-score standardization option
                show_zscore = st.toggle("Use z-score standardization", value=False,
                    help="Standardize expression values to highlight relative differences")

            # Generate button - placed OUTSIDE container to match volcano structure
            generate_button = st.button("Generate Heatmap")

            # Data generation and plot
            if generate_button:
                with st.spinner("Generating heatmap..."):
                    # Get structure from session state
                    structure = {"replicates": group_selections}

                    # Create heatmaps
                    detailed_heatmap, group_average_heatmap, plot_data, plot_data_means, scores = Visualizer.create_heatmap(
                        df,
                        group_selections,
                        structure,
                        n_proteins=n_proteins,
                        cluster_rows=cluster_rows,
                        cluster_cols=cluster_cols,
                        show_zscore=show_zscore,
                        use_original_for_stats=True
                    )

                    top_proteins = plot_data.index

                    # Prepare enhanced CSV with protein info and stats
                    csv_data = pd.DataFrame(index=top_proteins)

                    # Add protein names/IDs
                    if 'PG.Genes' in df.columns:
                        csv_data['Protein_Name'] = df.loc[top_proteins, 'PG.Genes']
                    elif 'Gene Name' in df.columns:
                        csv_data['Protein_Name'] = df.loc[top_proteins, 'Gene Name']
                    else:
                        csv_data['Protein_Name'] = top_proteins

                    # Add sample-level data and group means
                    for col in plot_data.columns:
                        csv_data[f'Sample_{col}'] = plot_data[col]
                    for group in plot_data_means.columns:
                        csv_data[f'Mean_{group}'] = plot_data_means[group]

                    # Add pairwise statistics
                    from itertools import combinations
                    for group1, group2 in combinations(group_selections.keys(), 2):
                        # Log2 fold change with epsilon to prevent division by zero
                        epsilon = 1e-10
                        ratio = (plot_data_means[group2] + epsilon) / (plot_data_means[group1] + epsilon)
                        log2fc = np.log2(ratio)
                        csv_data[f'Log2FC_{group2}_vs_{group1}'] = log2fc

                        # Calculate p-values
                        p_values = []
                        for protein in top_proteins:
                            group1_vals = plot_data[group_selections[group1]].loc[protein].dropna()
                            group2_vals = plot_data[group_selections[group2]].loc[protein].dropna()

                            if len(group1_vals) > 1 and len(group2_vals) > 1:
                                _, p_val = stats.ttest_ind(group1_vals, group2_vals, equal_var=False)
                                p_values.append(p_val)
                            else:
                                p_values.append(np.nan)

                        csv_data[f'pvalue_{group2}_vs_{group1}'] = p_values

                    # Add ANOVA statistics
                    max_fold_changes, f_stats, p_values = [], [], []
                    for protein in top_proteins:
                        protein_means, group_data = [], []
                        for group in group_selections:
                            group_cols = [col for col in structure["replicates"][group]]
                            group_vals = plot_data.loc[protein, group_cols].values
                            protein_means.append(np.mean(group_vals))
                            group_data.append(group_vals)

                        # Max fold change
                        fold_changes = []
                        for i in range(len(protein_means)):
                            for j in range(i+1, len(protein_means)):
                                if protein_means[i] > 0 and protein_means[j] > 0:
                                    fold_change = abs(np.log2(protein_means[j] / protein_means[i]))
                                    fold_changes.append(fold_change)

                        # ANOVA
                        try:
                            if all(len(g) >= 2 for g in group_data):
                                f_stat, p_val = stats.f_oneway(*group_data)
                                f_stats.append(f_stat)
                                p_values.append(p_val)
                            else:
                                f_stats.append(np.nan)
                                p_values.append(np.nan)
                        except:
                            f_stats.append(np.nan)
                            p_values.append(np.nan)

                        max_fold_changes.append(max(fold_changes) if fold_changes else np.nan)

                    # Add statistics to csv_data
                    csv_data['Max_Fold_Change'] = max_fold_changes
                    csv_data['F_Statistic'] = f_stats
                    csv_data['ANOVA_pvalue'] = p_values

                    # Generate CSV buffer
                    csv_buffer = csv_data.to_csv(index=True)

                    # Generate SVG data for both heatmaps
                    detailed_svg = Visualizer.get_matplotlib_svg(detailed_heatmap)
                    group_svg = Visualizer.get_matplotlib_svg(group_average_heatmap)

                    # Store everything in session state
                    st.session_state.heatmap_fig_detailed = detailed_heatmap 
                    st.session_state.heatmap_fig_group = group_average_heatmap
                    st.session_state.heatmap_csv_data = csv_buffer
                    st.session_state.heatmap_detailed_svg = detailed_svg
                    st.session_state.heatmap_group_svg = group_svg

            # Display heatmaps from session state
            with heatmap_display_container:
                if 'heatmap_fig_detailed' in st.session_state:
                    st.write("### Detailed Sample Heatmap")
                    st.pyplot(st.session_state.heatmap_fig_detailed.figure)

                if 'heatmap_fig_group' in st.session_state:
                    st.write("### Group Average Heatmap")
                    st.pyplot(st.session_state.heatmap_fig_group.figure)

            # Add download options in a separate container
            if 'heatmap_fig_detailed' in st.session_state:
                download_container = st.container()
                with download_container:
                    st.write("### Download Options")
                    col1, col2, col3 = st.columns(3)

                    # CSV download
                    with col1:
                        st.download_button(
                            "Download Data Table (CSV)",
                            st.session_state.heatmap_csv_data,
                            file_name="heatmap_data.csv",
                            mime="text/csv",
                            key="download_heatmap_csv"
                        )

                    # SVG downloads
                    with col2:
                        st.download_button(
                            "Download Detailed Heatmap (SVG)",
                            st.session_state.heatmap_detailed_svg,
                            file_name="detailed_heatmap.svg",
                            mime="image/svg+xml",
                            key="download_detailed_heatmap"
                        )

                    with col3:
                        st.download_button(
                            "Download Group Average Heatmap (SVG)",
                            st.session_state.heatmap_group_svg,
                            file_name="group_average_heatmap.svg",
                            mime="image/svg+xml",
                            key="download_group_heatmap"
                        )

    elif plot_type == "Custom Protein Heatmap":
        st.subheader("Custom Protein Heatmap")

        # Get group selections
        group_selections = st.session_state.get('group_selections', {})

        if not group_selections:
            st.warning("Please define sample groups in the Data Upload page first.")
        else:
            # Create a container for the heatmap to persist
            custom_heatmap_display = st.container()

            # Input and settings container
            settings_container = st.container()
            with settings_container:
                # Input for protein list
                protein_input = st.text_area(
                    "Enter protein names (one per line):",
                    help="Enter the names of proteins you want to visualize in the heatmap."
                )

                if protein_input:
                    protein_list = [p.strip() for p in protein_input.split('\n') if p.strip()]

                    # Clustering options
                    col1, col2 = st.columns(2)
                    with col1:
                        cluster_rows = st.toggle("Cluster proteins", value=True)
                    with col2:
                        cluster_cols = st.toggle("Cluster samples", value=True)

                    # Add note about statistical results
                    st.info("Statistical comparisons are available in the downloadable results table.")

                    # Z-score standardization option
                    show_zscore = st.toggle("Use z-score standardization", value=True,
                        help="Standardize expression values to highlight relative differences")

            # Generate button - placed OUTSIDE container to match volcano structure
            generate_button = st.button("Generate Custom Heatmap")

            # Data generation and plot
            if generate_button and protein_input:
                with st.spinner("Generating custom heatmap..."):
                        # Get structure from session state
                        structure = {"replicates": group_selections}

                        # Create heatmaps
                        detailed_heatmap, group_average_heatmap, plot_data, plot_data_means, scores = Visualizer.create_custom_heatmap(
                            df,
                            protein_list,
                            group_selections,
                            structure,
                            cluster_rows=cluster_rows,
                            cluster_cols=cluster_cols,
                            show_zscore=show_zscore
                        )

                        top_proteins = plot_data.index

                        # Prepare enhanced CSV with protein info and stats
                        csv_data = pd.DataFrame(index=top_proteins)

                        # Add protein names/IDs
                        if 'PG.Genes' in df.columns:
                            csv_data['Protein_Name'] = df.loc[top_proteins, 'PG.Genes']
                        elif 'Gene Name' in df.columns:
                            csv_data['Protein_Name'] = df.loc[top_proteins, 'Gene Name']
                        else:
                            csv_data['Protein_Name'] = top_proteins

                        # Add sample-level data
                        for col in plot_data.columns:
                            csv_data[f'Sample_{col}'] = plot_data[col]

                        # Add group means
                        for group in plot_data_means.columns:
                            csv_data[f'Mean_{group}'] = plot_data_means[group]

                        # Add pairwise statistics
                        from itertools import combinations
                        for group1, group2 in combinations(group_selections.keys(), 2):
                            # Calculate log2 fold change with epsilon to prevent division by zero
                            epsilon = 1e-10
                            ratio = (plot_data_means[group2] + epsilon) / (plot_data_means[group1] + epsilon)
                            log2fc = np.log2(ratio)
                            csv_data[f'Log2FC_{group2}_vs_{group1}'] = log2fc

                            # Calculate p-values
                            p_values = []
                            for protein in top_proteins:
                                group1_vals = plot_data[group_selections[group1]].loc[protein].dropna()
                                group2_vals = plot_data[group_selections[group2]].loc[protein].dropna()

                                if len(group1_vals) > 1 and len(group2_vals) > 1:
                                    _, p_val = stats.ttest_ind(group1_vals, group2_vals, equal_var=False)
                                    p_values.append(p_val)
                                else:
                                    p_values.append(np.nan)

                            csv_data[f'pvalue_{group2}_vs_{group1}'] = p_values

                        # Calculate max fold change directly
                        max_fold_changes = []
                        f_stats = []
                        p_values = []

                        for protein in top_proteins:
                            protein_means = []
                            group_data = []

                            for group in group_selections:
                                group_cols = [col for col in structure["replicates"][group]]
                                group_vals = plot_data.loc[protein, group_cols].values
                                group_mean = np.mean(group_vals)
                                protein_means.append(group_mean)
                                group_data.append(group_vals)

                            # Calculate max fold change between any two groups
                            fold_changes = []
                            for i in range(len(protein_means)):
                                for j in range(i+1, len(protein_means)):
                                    if protein_means[i] > 0 and protein_means[j] > 0:
                                        fold_change = abs(np.log2(protein_means[j] / protein_means[i]))
                                        fold_changes.append(fold_change)

                            # Calculate ANOVA statistics
                            try:
                                if all(len(g) >= 2 for g in group_data):
                                    f_stat, p_val = stats.f_oneway(*group_data)
                                    f_stats.append(f_stat)
                                    p_values.append(p_val)
                                else:
                                    f_stats.append(np.nan)
                                    p_values.append(np.nan)
                            except:
                                f_stats.append(np.nan)
                                p_values.append(np.nan)

                            max_fold_changes.append(max(fold_changes) if fold_changes else np.nan)

                        # Add statistics to csv_data
                        csv_data['Max_Fold_Change'] = max_fold_changes
                        csv_data['F_Statistic'] = f_stats
                        csv_data['ANOVA_pvalue'] = p_values

                        csv_buffer = csv_data.to_csv(index=True)

                        if detailed_heatmap is not None:
                            # Generate SVG data for both heatmaps
                            detailed_svg = Visualizer.get_matplotlib_svg(detailed_heatmap)
                            group_svg = Visualizer.get_matplotlib_svg(group_average_heatmap)

                            # Store everything directly in session state with simpler structure
                            st.session_state.custom_heatmap_fig_detailed = detailed_heatmap
                            st.session_state.custom_heatmap_fig_group = group_average_heatmap
                            st.session_state.custom_heatmap_csv_data = csv_buffer
                            st.session_state.custom_heatmap_detailed_svg = detailed_svg
                            st.session_state.custom_heatmap_group_svg = group_svg
                        else:
                            st.error("No matching proteins found in the dataset.")

            # Display heatmaps from session state
            with custom_heatmap_display:
                if 'custom_heatmap_fig_detailed' in st.session_state:
                    st.write("### Detailed Sample Heatmap")
                    st.pyplot(st.session_state.custom_heatmap_fig_detailed.figure)

                if 'custom_heatmap_fig_group' in st.session_state:
                    st.write("### Group Average Heatmap")
                    st.pyplot(st.session_state.custom_heatmap_fig_group.figure)

            # Add download options in a separate container
            if 'custom_heatmap_fig_detailed' in st.session_state:
                custom_download_container = st.container()
                with custom_download_container:
                    st.write("### Download Options")
                    col1, col2, col3 = st.columns(3)

                    # Downloads with direct session state variables
                    with col1:
                        st.download_button(
                            "Download Data Table (CSV)",
                            st.session_state.custom_heatmap_csv_data,
                            file_name="custom_heatmap_data.csv",
                            mime="text/csv",
                            key="download_custom_data_csv"
                        )

                    with col2:
                        st.download_button(
                            "Download Detailed Heatmap (SVG)",
                            st.session_state.custom_heatmap_detailed_svg,
                            file_name="custom_detailed_heatmap.svg",
                            mime="image/svg+xml",
                            key="download_custom_detailed_heatmap"
                        )

                    with col3:
                        st.download_button(
                            "Download Group Average Heatmap (SVG)",
                            st.session_state.custom_heatmap_group_svg,
                            file_name="custom_group_average_heatmap.svg",
                            mime="image/svg+xml",
                            key="download_custom_group_heatmap"
                        )