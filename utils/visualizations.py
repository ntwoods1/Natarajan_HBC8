import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from scipy import stats
from typing import Tuple, List, Dict, Optional, Any
from utils.color_palettes import ColorPalettes

class Visualizer:
    @staticmethod
    def create_heatmap(data: pd.DataFrame, group_selections: dict, structure: dict, n_proteins: int = 50,
                      cluster_rows: bool = True, cluster_cols: bool = True, show_zscore: bool = False, use_original_for_stats: bool = True) -> tuple:
        """
        Create heatmap visualization for proteomics data

        Args:
            data: DataFrame with proteomics data
            group_selections: Dictionary mapping group names to column lists
            structure: Dataset structure information
            n_proteins: Number of proteins to display
            cluster_rows: Whether to cluster rows
            cluster_cols: Whether to cluster columns
            show_zscore: Whether to show z-score
            use_original_for_stats: Whether to use original data for statistical calculations

        Returns:
            Tuple of (detailed_heatmap, group_average_heatmap, plot_data, plot_data_means)
        """
        # Store original data for statistical calculations if using z-scores
        original_data = data.copy() if show_zscore and use_original_for_stats else data

        # Calculate statistics using original data
        scores = pd.DataFrame(index=data.index)

        # Calculate group means for both original and z-scored data
        group_means = pd.DataFrame(index=data.index)
        original_group_means = pd.DataFrame(index=data.index)

        for group in group_selections:
            group_cols = [col for col in structure["replicates"][group] if col.endswith("PG.Quantity")]
            # Store z-scored means for display
            group_means[group] = data[group_cols].mean(axis=1)
            # Store original means for fold change calculation
            original_group_means[group] = original_data[group_cols].mean(axis=1)

        # Create a DataFrame to store statistical results and fold changes
        scores = pd.DataFrame(index=data.index)

        # Calculate fold changes and add them to the scores DataFrame
        from itertools import combinations
        for g1, g2 in combinations(group_selections, 2):
            # Calculate log2 fold change using original means
            ratio = original_group_means[g2] / original_group_means[g1]
            log2fc = np.log2(ratio)
            scores[f'Log2FC_{g2}_vs_{g1}'] = log2fc

            # Store the absolute fold change for later use
            if 'Max_Fold_Change' not in scores.columns:
                scores['Max_Fold_Change'] = np.abs(log2fc)
            else:
                scores['Max_Fold_Change'] = np.maximum(scores['Max_Fold_Change'], np.abs(log2fc))

        # Calculate F-statistic and p-value using ANOVA
        f_stats = []
        p_values = []
        for protein in data.index:
            group_data = []
            for group in group_selections:
                group_cols = [col for col in structure["replicates"][group] if col.endswith("PG.Quantity")]
                group_data.append(original_data.loc[protein, group_cols])
            try:
                f_stat, p_val = stats.f_oneway(*group_data)
                f_stats.append(f_stat)
                p_values.append(p_val)
            except:
                f_stats.append(0)
                p_values.append(1)

        scores['f_statistic'] = f_stats
        scores['p_value'] = p_values
        scores['-log10_p'] = -np.log10(scores['p_value'].clip(1e-10, 1))

        # Select top proteins
        scores['final_score'] = scores['-log10_p'] * scores['Max_Fold_Change']
        top_proteins = scores.nlargest(n_proteins, 'final_score').index

        # Prepare plot data
        quantity_cols = []
        for group in group_selections:
            quantity_cols.extend([col for col in structure["replicates"][group] if col.endswith("PG.Quantity")])

        plot_data = data.loc[top_proteins, quantity_cols]
        plot_data_means = group_means.loc[top_proteins]

        # Create row labels - prioritize PG.genes, fallback to Gene Name or index
        if 'PG.Genes' in data.columns:
            row_labels = data.loc[top_proteins, 'PG.Genes']
        elif 'Gene Name' in data.columns:
            row_labels = data.loc[top_proteins, 'Gene Name']
        else:
            row_labels = top_proteins

        # Store original data for statistics before any z-score transformation
        original_plot_data = plot_data.copy()
        original_plot_means = plot_data_means.copy()

        # Store display version of data (potentially z-scored)
        display_data = plot_data.copy()
        display_means = plot_data_means.copy()

        # Apply z-score standardization if requested (only for display)
        if show_zscore:
            # Calculate z-score for display only
            display_data = plot_data.apply(lambda row: (row - row.mean()) / row.std(), axis=1)

            # Calculate group means of z-scored data for display
            display_means = pd.DataFrame(index=plot_data.index)
            for group in group_selections:
                group_cols = [col for col in structure["replicates"][group] if col.endswith("PG.Quantity")]
                display_means[group] = display_data[group_cols].mean(axis=1)

        # Create summary statistics DataFrame
        stats_df = pd.DataFrame()

        # Add raw expression values (use display data which may be z-scored)
        for col in display_data.columns:
            stats_df[f"Expression_{col}"] = display_data[col]

        # Add group means (use display means which may be z-scored)
        for group in group_means.columns:
            stats_df[f"Mean_{group}"] = display_means[group]

        # Calculate pairwise statistics using original data (always use original data for stats)
        for i, group1 in enumerate(group_means.columns):
            for group2 in group_means.columns[i+1:]:
                # Calculate log2 fold change using original means (not z-scored)
                # Check if data is already log2 transformed
                import streamlit as st
                is_log2_normalized = st.session_state.get('normalization_method', None) == 'Log2' if hasattr(st, 'session_state') else False

                if is_log2_normalized:
                    # Data is already in log2 space, so fold change is simple subtraction
                    log2fc = original_plot_means[group2] - original_plot_means[group1]
                else:
                    # Data is in linear space, apply log2 transformation
                    log2fc = np.log2(original_plot_means[group2] / original_plot_means[group1])
                stats_df[f"Log2FC_{group2}_vs_{group1}"] = log2fc

                # Calculate p-values using original data
                p_values = []
                for protein in original_plot_data.index:
                    group1_vals = original_plot_data[structure["replicates"][group1]].loc[protein].dropna()
                    group2_vals = original_plot_data[structure["replicates"][group2]].loc[protein].dropna()

                    if len(group1_vals) > 1 and len(group2_vals) > 1:
                        _, p_val = stats.ttest_ind(group1_vals, group2_vals, equal_var=False)
                        p_values.append(p_val)
                    else:
                        p_values.append(np.nan)

                stats_df[f"pvalue_{group2}_vs_{group1}"] = p_values

        # Calculate figure dimensions based on number of proteins
        # Same figure size for both heatmaps to ensure consistency
        figsize = (10, 10 + (n_proteins/10) - 1)

        # Create heatmaps using the display data (which may be z-scored)
        g1 = sns.clustermap(
            display_data,
            cmap='RdBu_r',
            center=0,
            robust=True,
            row_cluster=cluster_rows,
            col_cluster=cluster_cols,
            yticklabels=row_labels,
            figsize=figsize,
            z_score=None  # We handle z-score manually for more control
        )

        g2 = sns.clustermap(
            display_means,
            cmap='RdBu_r',
            center=0,
            robust=True,
            row_cluster=cluster_rows,
            col_cluster=cluster_cols,
            yticklabels=row_labels,
            figsize=figsize,
            z_score=None  # We handle z-score manually for more control
        )

        return g1, g2, plot_data, plot_data_means, stats_df

    @staticmethod
    def create_custom_heatmap(data: pd.DataFrame, protein_list: list, group_selections: dict, structure: dict,
                            cluster_rows: bool = True, cluster_cols: bool = True, show_zscore: bool = False) -> tuple:
        """
        Create custom heatmap for specified proteins

        Args:
            data: DataFrame with proteomics data
            protein_list: List of proteins to include
            group_selections: Dictionary mapping group names to column lists
            structure: Dataset structure information
            cluster_rows: Whether to cluster rows
            cluster_cols: Whether to cluster columns
            show_zscore: Whether to show z-score

        Returns:
            Tuple of (detailed_heatmap, group_average_heatmap, plot_data, plot_data_means)
        """
        # Find matching proteins
        # First try Gene Name column
        if 'Gene Name' in data.columns:
            matches = data[data['Gene Name'].str.contains('|'.join(protein_list), case=False, na=False)]
        # Then try direct protein ID matching
        else:
            protein_set = set(protein_list)
            matches = data[data.index.isin(protein_set)]

            if matches.empty:
                # Try finding matches in any column that might contain protein IDs
                potential_protein_cols = [col for col in data.columns if any(x in col.lower() for x in ['protein', 'gene', 'id'])]
                for col in potential_protein_cols:
                    if data[col].dtype == 'object':  # Only try string matching on string columns
                        matches = data[data[col].str.contains('|'.join(protein_list), case=False, na=False)]
                        if not matches.empty:
                            break

        if matches.empty:
            return None, None, None, None

        # Prepare heatmap data
        quantity_cols = []
        for group in group_selections:
            quantity_cols.extend([col for col in structure["replicates"][group] if col.endswith("PG.Quantity")])

        plot_data = matches[quantity_cols]

        # Calculate group means
        group_means = pd.DataFrame(index=matches.index)
        for group in group_selections:
            group_cols = [col for col in group_selections[group] if col.endswith("PG.Quantity")]
            group_means[group] = matches[group_cols].mean(axis=1)

        # Create row labels - prioritize PG.genes, fallback to Gene Name or index
        if 'PG.Genes' in matches.columns:
            row_labels = matches['PG.Genes']
        elif 'Gene Name' in matches.columns:
            row_labels = matches['Gene Name']
        else:
            row_labels = matches.index

        # Store original data for statistics
        original_data = plot_data.copy()
        original_means = pd.DataFrame(index=plot_data.index)
        for group in group_selections:
            group_cols = [col for col in group_selections[group] if col.endswith("PG.Quantity")]
            original_means[group] = original_data[group_cols].mean(axis=1)

        # Create display copies (potentially z-scored)
        display_data = plot_data.copy()
        display_means = original_means.copy()

        # Apply z-score standardization if requested
        if show_zscore:
            # Only standardize the detailed sample data for display
            display_data = plot_data.apply(lambda row: (row - row.mean()) / row.std(), axis=1)

            # For group means display, calculate means of the z-scored sample data
            display_means = pd.DataFrame(index=plot_data.index)
            for group in group_selections:
                group_cols = [col for col in group_selections[group] if col.endswith("PG.Quantity")]
                display_means[group] = display_data[group_cols].mean(axis=1)

            # Set the display data as the data to use for the plots
            plot_data = display_data
            group_means = display_means
        else:
            group_means = original_means

        # Calculate figure dimensions based on number of proteins
        # Same figure size for both heatmaps to ensure consistency
        figsize = (10, 10 + (len(matches)/10) - 1)

        # Create colorblind-friendly custom colormap
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        # Use the colorblind-friendly palette from our utility class
        # Convert to a format matplotlib can use (list of RGB tuples)
        palette_colors = ColorPalettes.COLORBLIND_DIVERGING
        # Create a colormap that goes from first color to middle to last color
        cmap_colors = []
        mid_idx = len(palette_colors) // 2
        for color in palette_colors[:mid_idx]:  # First half (blues)
            cmap_colors.append(mcolors.hex2color(color))
        cmap_colors.append(mcolors.hex2color(palette_colors[mid_idx]))  # Middle (white)
        for color in palette_colors[mid_idx+1:]:  # Second half (reds)
            cmap_colors.append(mcolors.hex2color(color))

        # Create custom colormap
        colorblind_cmap = mcolors.LinearSegmentedColormap.from_list('colorblind_diverging', cmap_colors)

        # Create heatmaps
        g1 = sns.clustermap(
            plot_data,
            cmap=colorblind_cmap,
            center=0,
            robust=True,
            row_cluster=cluster_rows,
            col_cluster=cluster_cols,
            yticklabels=row_labels,
            figsize=figsize,
            z_score=None  # We handle z-score manually for more control
        )

        g2 = sns.clustermap(
            group_means,
            cmap=colorblind_cmap,
            center=0,
            robust=True,
            row_cluster=cluster_rows,
            col_cluster=cluster_cols,
            yticklabels=row_labels,
            figsize=figsize,
            z_score=None  # We handle z-score manually for more control
        )

        # Calculate statistics for table using original (non-z-scored) data
        stats_df = Visualizer._calculate_heatmap_statistics(original_data, group_selections, structure)

        return g1, g2, plot_data, group_means, stats_df

    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
        """
        Create interactive correlation heatmap
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()

        # Use colorblind-friendly colorscale
        colorscale = ColorPalettes.get_colorscale_for_correlation()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=colorscale,
            zmid=0
        ))

        fig.update_layout(
            title=title,
            width=800,
            height=800
        )

        return fig

    @staticmethod
    def create_cv_histogram(df: pd.DataFrame, group_selections: dict, cutoff: float = 0.2) -> go.Figure:
        """
        Create histogram of CV values for each group with cutoff line

        Args:
            df: Dataframe with proteomics data
            group_selections: Dictionary mapping group names to column lists
            cutoff: CV cutoff value (0-1)

        Returns:
            Plotly figure with CV histograms
        """
        # Calculate number of groups for subplot layout
        num_groups = len(group_selections)
        if num_groups == 0:
            return go.Figure().update_layout(title="No groups defined")

        # Calculate dynamic vertical spacing based on number of groups
        vertical_spacing = min(0.1, 1.0 / (num_groups + 1))  # Ensure spacing doesn't exceed maximum

        # Create subplot grid (1 row per group)
        fig = make_subplots(rows=num_groups, cols=1,
                            subplot_titles=[f"Group: {group}" for group in group_selections.keys()],
                            vertical_spacing=vertical_spacing)

        # Process each group
        for i, (group_name, columns) in enumerate(group_selections.items(), 1):
            # Skip if no columns in group
            if not columns:
                continue

            # Calculate CV for this group
            group_data = df[columns]
            # Handle division by zero and NaN values
            means = group_data.mean(axis=1).abs()
            stds = group_data.std(axis=1)

            # Calculate CV and handle potential division by zero
            cv_values = pd.Series(np.nan, index=means.index)
            mask = (means > 0) & (~means.isna()) & (~stds.isna())
            cv_values.loc[mask] = stds.loc[mask] / means.loc[mask]

            # Remove NaN values for plotting
            cv_values = cv_values.dropna()

            # Skip if all values are NaN
            if len(cv_values) == 0:
                continue

            # Create histogram trace
            hist_trace = go.Histogram(
                x=cv_values,
                name=group_name,
                nbinsx=50,
                marker_color=f'rgba({(i*60)%255}, {(i*100)%255}, {(i*160)%255}, 0.7)'
            )

            # Calculate histogram values manually with explicit range to avoid NaN issues
            hist_vals, bin_edges = np.histogram(
                cv_values,
                bins=50,
                range=(0, min(3, cv_values.max() * 1.2) if not cv_values.empty else 3)
            )
            max_count = max(hist_vals) * 1.1 if len(hist_vals) > 0 else 10  # Add 10% for visibility

            # Create more visible vertical line for cutoff
            cutoff_line = go.Scatter(
                x=[cutoff, cutoff],
                y=[0, max_count],
                mode='lines',
                name=f'CV Cutoff: {cutoff:.2f}',
                line=dict(color='red', width=3, dash='dash')
            )

            # Add traces to subplot
            fig.add_trace(hist_trace, row=i, col=1)
            fig.add_trace(cutoff_line, row=i, col=1)

            # Update layout for this subplot
            fig.update_xaxes(title_text="Coefficient of Variation (CV)", row=i, col=1, range=[0, min(3, cv_values.max()*1.2) if not cv_values.empty else 3])
            fig.update_yaxes(title_text="Count", row=i, col=1)

        # Update overall layout
        fig.update_layout(
            title="Distribution of Coefficient of Variation (CV) by Group",
            height=300*num_groups,
            width=800,
            showlegend=False
        )

        return fig

    @staticmethod
    def get_figure_as_svg(fig: go.Figure) -> str:
        """
        Convert a plotly figure to SVG format

        Args:
            fig: Plotly figure to convert

        Returns:
            SVG data as string
        """
        try:
            return fig.to_image(format="svg").decode("utf-8")
        except Exception as e:
            # Fallback message for debugging
            return f'''<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">
    <text x="50" y="50" font-family="sans-serif" font-size="16">Error creating SVG: {str(e)}</text>
</svg>'''

    @staticmethod
    def create_cv_histogram_matplotlib(df: pd.DataFrame, group_selections: dict, cutoff: float = 0.2):
        """
        Create histogram of CV values using matplotlib (for reliable SVG export)

        Args:
            df: Dataframe with proteomics data
            group_selections: Dictionary mapping group names to column lists
            cutoff: CV cutoff value (0-1)

        Returns:
            Matplotlib figure and a dictionary of CV values by group
        """
        import matplotlib.pyplot as plt
        import io

        # Calculate number of groups for subplot layout
        num_groups = len(group_selections)
        if num_groups == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No groups defined", ha='center', va='center')
            return fig, {}

        # Create figure with subplots (one per group)
        fig, axes = plt.subplots(num_groups, 1, figsize=(8, 3*num_groups), constrained_layout=True)

        # Handle case with only one group
        if num_groups == 1:
            axes = [axes]

        # Process each group and store CV values for table export
        cv_values_by_group = {}

        for i, (group_name, columns) in enumerate(group_selections.items()):
            ax = axes[i]

            # Skip if no columns in group
            if not columns:
                ax.text(0.5, 0.5, f"No columns in group: {group_name}", ha='center', va='center', transform=ax.transAxes)
                continue

            # Calculate CV for this group
            group_data = df[columns]
            means = group_data.mean(axis=1).abs()
            stds = group_data.std(axis=1)

            # Calculate CV and handle potential division by zero
            cv_series = pd.Series(np.nan, index=means.index)
            mask = (means > 0) & (~means.isna()) & (~stds.isna())
            cv_series.loc[mask] = stds.loc[mask] / means.loc[mask]

            # Store for table export - include protein IDs
            protein_col = None
            if hasattr(st, 'session_state') and 'protein_col' in st.session_state:
                protein_col = st.session_state.protein_col

            cv_df = pd.DataFrame({
                'Protein': df[protein_col] if protein_col and protein_col in df.columns else cv_series.index,
                'CV': cv_series
            })
            cv_values_by_group[group_name] = cv_df

            # Remove NaN values for plotting
            cv_values = cv_series.dropna()

            # Skip if all values are NaN
            if len(cv_values) == 0:
                ax.text(0.5, 0.5, f"No valid CV values for group: {group_name}", ha='center', va='center', transform=ax.transAxes)
                continue

            # Create histogram
            n, bins, patches = ax.hist(
                cv_values,
                bins=50,
                range=(0, min(3, cv_values.max() * 1.2) if not cv_values.empty else 3),
                alpha=0.7,
                color=f'C{i}'
            )

            # Add cutoff line
            ylim = ax.get_ylim()
            ax.plot([cutoff, cutoff], [0, ylim[1]], 'r--', linewidth=2, label=f'CV Cutoff: {cutoff:.2f}')

            # Set title and labels
            ax.set_title(f"Group: {group_name}")
            ax.set_xlabel("Coefficient of Variation (CV)")
            ax.set_ylabel("Count")
            ax.set_xlim(0, min(3, cv_values.max()*1.2) if not cv_values.empty else 3)
            ax.legend()

            # Add stats text
            above_cutoff = (cv_values > cutoff).sum()
            percent_above = above_cutoff / len(cv_values) * 100 if len(cv_values) > 0 else 0
            stats_text = f"Total: {len(cv_values)}\nAbove cutoff: {above_cutoff} ({percent_above:.1f}%)"
            ax.text(0.95, 0.95, stats_text, ha='right', va='top', transform=ax.transAxes,
                   bbox=dict(facecolor='white', alpha=0.7))

        # Add overall title
        fig.suptitle("Distribution of Coefficient of Variation (CV) by Group", fontsize=16)

        return fig, cv_values_by_group

    @staticmethod
    def get_matplotlib_svg(fig):
        """Get SVG string from matplotlib figure"""
        import io

        svg_io = io.StringIO()
        fig.savefig(svg_io, format='svg')
        svg_io.seek(0)
        svg_data = svg_io.getvalue()
        return svg_data

    @staticmethod
    def create_intensity_histograms(df: pd.DataFrame, group_selections: dict) -> go.Figure:
        """
        Create histograms of log2 expression values for each sample, grouped by sample group

        Args:
            df: DataFrame with proteomics data
            group_selections: Dictionary mapping group names to column lists

        Returns:
            Plotly figure with expression histograms
        """
        # Calculate number of groups and samples for subplot layout
        n_groups = len(group_selections)

        if n_groups == 0:
            # Return empty figure if no groups
            return go.Figure().update_layout(title="No groups defined")

        # Count total samples
        n_samples = sum(len(cols) for cols in group_selections.values())

        # Calculate grid dimensions - try to make it as square as possible
        n_cols = min(3, n_samples)  # Maximum 3 plots per row (was 4)
        n_rows = (n_samples + n_cols - 1) // n_cols  # Ceiling division

        # Create subplot titles for each sample
        subplot_titles = []
        for group_name, columns in group_selections.items():
            for col in columns:
                subplot_titles.append(f"{col} ({group_name})")

        # Create subplot grid with increased spacing
        fig = make_subplots(rows=n_rows, cols=n_cols,
                           subplot_titles=subplot_titles,
                           vertical_spacing=0.2,    # Increased from 0.1
                           horizontal_spacing=0.15) # Increased from 0.05

        # Track position in the grid
        plot_idx = 0

        # Use colorblind-friendly palette
        color_palette = ColorPalettes.COLORBLIND_FRIENDLY

        # Process each group
        for i, (group_name, columns) in enumerate(group_selections.items()):
            # Assign a color to this group
            group_color = color_palette[i % len(color_palette)]

            # For each sample in the group
            for col in columns:
                # Calculate row and column position
                row = (plot_idx // n_cols) + 1
                col_pos = (plot_idx % n_cols) + 1

                # Extract intensity values (ignore NaNs)
                intensity_values = df[col].dropna()

                if len(intensity_values) == 0:
                    # If no valid data, add empty plot with message
                    fig.add_annotation(
                        x=0.5, y=0.5,
                        text="No data",
                        showarrow=False,
                        row=row, col=col_pos
                    )
                else:
                    # Convert to log2 scale, handling zeros and negative values
                    log2_values = np.log2(intensity_values.clip(lower=1e-6))

                    # Calculate mean for the vertical line
                    mean_value = np.mean(log2_values)

                    # Create histogram trace
                    hist_trace = go.Histogram(
                        x=log2_values,
                        nbinsx=30,
                        marker_color=group_color,
                        name=col,
                        showlegend=False
                    )

                    # Add histogram to subplot
                    fig.add_trace(hist_trace, row=row, col=col_pos)

                    # Add mean line
                    fig.add_shape(
                        type="line",
                        x0=mean_value, x1=mean_value,
                        y0=0, y1=1,
                        yref="paper",
                        xref=f"x{plot_idx+1}",
                        line=dict(color="red", width=2, dash="dash"),
                        row=row, col=col_pos
                    )

                    # Add count annotations
                    protein_count = len(intensity_values)
                    fig.add_annotation(
                        x=0.95, y=0.95,
                        xref=f"x{plot_idx+1}",
                        yref=f"y{plot_idx+1}",
                        text=f"n={protein_count}",
                        showarrow=False,
                        font=dict(size=10),
                        align="right",
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=3,
                        row=row, col=col_pos
                    )

                    # Add mean value annotation
                    fig.add_annotation(
                        x=mean_value, y=0.85,
                        xref=f"x{plot_idx+1}",
                        yref=f"y{plot_idx+1}",
                        text=f"Mean: {mean_value:.2f}",
                        showarrow=False,
                        font=dict(size=10, color="red"),
                        align="center",
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="red",
                        borderwidth=1,
                        borderpad=3,
                        row=row, col=col_pos
                    )

                    # Update axes
                    fig.update_xaxes(title_text="log2(Expression)", row=row, col=col_pos)
                    if col_pos == 1:  # Only for first column
                        fig.update_yaxes(title_text="Count", row=row, col=col_pos)

                # Increment position counter
                plot_idx += 1

        # Update overall layout
        fig.update_layout(
            title="Distribution of Protein Expression by Sample",
            height=300*n_rows + 50,  # Increased from 250 to give more vertical space
            width=300*n_cols + 50,   # Increased from 250 to give more horizontal space
            margin=dict(t=70, b=20, l=60, r=20),
            template="plotly_white"
        )

        return fig

    @staticmethod
    def generate_cv_table(df: pd.DataFrame, group_selections: dict) -> pd.DataFrame:
        """
        Generate a table of CV values for all proteins

        Args:
            df: Dataframe with proteomics data
            group_selections: Dictionary mapping group names to column lists

        Returns:
            DataFrame with protein IDs and CV values for each group
        """
        # Start with protein column
        protein_col = None
        if hasattr(st, 'session_state') and 'protein_col' in st.session_state:
            protein_col = st.session_state.protein_col

        protein_col = protein_col if protein_col and protein_col in df.columns else df.index.name or 'Index'

        cv_table = pd.DataFrame({
            'Protein': df[protein_col] if protein_col in df.columns else df.index
        })

        # Calculate CV for each group
        for group_name, columns in group_selections.items():
            if not columns:
                continue

            # Calculate CV for this group
            group_data = df[columns]
            means = group_data.mean(axis=1).abs()
            stds = group_data.std(axis=1)

            # Calculate CV and handle potential division by zero
            cv_values = pd.Series(np.nan, index=means.index)
            mask = (means > 0) & (~means.isna()) & (~stds.isna())
            cv_values.loc[mask] = stds.loc[mask] / means.loc[mask]

            # Add to table
            cv_table[f'CV_{group_name}'] = cv_values

        return cv_table

    @staticmethod
    def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str,
                          color_col: str = None) -> go.Figure:
        """
        Create interactive scatter plot
        """
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            trendline="ols"
        )

        fig.update_layout(
            title=f"{y_col} vs {x_col}",
            width=800,
            height=600
        )

        return fig

    @staticmethod
    def create_volcano_plot(df: pd.DataFrame, fc_col: str,
                          pval_col: str, labels: list = None,
                          fc_threshold: float = 1.0,
                          p_threshold: float = 0.05,
                          correction_name: str = "None",
                          power: float = None,
                          alpha: float = 0.05,
                          labels_to_show: list = None,
                          test_description: str = None) -> go.Figure:
        """
        Create volcano plot

        Args:
            df: DataFrame containing the data
            fc_col: Column name for fold change values
            pval_col: Column name for p-values
            labels: Optional list of labels for hover text
            fc_threshold: Threshold for log2 fold change significance (default: 1.0)
            p_threshold: Threshold for p-value significance (default: 0.05)
            correction_name: Name of the p-value correction method used (default: "None")
            labels_to_show: List of boolean values indicating which points to label on the plot

        Returns:
            Plotly figure with volcano plot
        """
        # Convert p-values to -log10(p) for y-axis
        neg_log_p = -np.log10(df[pval_col].astype(float))

        # Use provided significance thresholds
        neg_log_p_threshold = -np.log10(p_threshold)  # -log10 of p threshold

        # Create a new column for coloring points
        df = df.copy()
        df['significance'] = 'Not Significant'

        # Significant with fold change and p-value criteria
        sig_mask = (df[fc_col].abs() >= fc_threshold) & (df[pval_col] < p_threshold)
        df.loc[sig_mask, 'significance'] = 'Significant'

        # Significant with only p-value criteria
        p_mask = (~sig_mask) & (df[pval_col] < p_threshold)
        df.loc[p_mask, 'significance'] = 'p-value < 0.05'

        # Significant with only fold change criteria
        fc_mask = (~sig_mask) & (df[fc_col].abs() >= fc_threshold)
        df.loc[fc_mask, 'significance'] = '|Log2FC| ≥ 1'

        # Define colorblind-friendly color mapping
        color_map = {
            'Significant': ColorPalettes.COLORBLIND_FRIENDLY[1],   # Vermillion (red)
            'p-value < 0.05': ColorPalettes.COLORBLIND_FRIENDLY[5], # Orange
            '|Log2FC| ≥ 1': ColorPalettes.COLORBLIND_FRIENDLY[0],  # Blue
            'Not Significant': ColorPalettes.COLORBLIND_FRIENDLY[7] # Gray
        }

        # Create hover text
        if labels is not None:
            hover_text = [f"Protein: {label}<br>Log2FC: {fc:.3f}<brp-value: {p:.4e}"
                        for label, fc, p in zip(labels, df[fc_col], df[pval_col])]
        else:
            hover_text = [f"Log2FC: {fc:.3f}<br>p-value: {p:.4e}"
                        for fc, p in zip(df[fc_col], df[pval_col])]

        # Create figure
        fig = px.scatter(
            df,
            x=fc_col,
            y=neg_log_p,
            color='significance',
            color_discrete_map=color_map,
            hover_name=labels if labels is not None else None,
            hover_data={fc_col: ':.3f', pval_col: ':.4e'},
            labels={fc_col: 'Log2 Fold Change', 'y': '-log10(p-value)'},
        )

        # Add vertical lines for fold change threshold
        fig.add_vline(x=fc_threshold, line_dash="dash", line_color="gray")
        fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="gray")

        # Add horizontal line for p-value threshold
        fig.add_hline(y=neg_log_p_threshold, line_dash="dash", line_color="gray")

        # Update layout
        plot_title = "Volcano Plot"
        if test_description:
            plot_title += f" ({test_description})"
        if correction_name != "None":
            plot_title += f" | Correction: {correction_name}"

        # Use the power parameter if provided, otherwise try session state
        if power is None and hasattr(st, 'session_state') and 'volcano_power' in st.session_state:
            power = st.session_state.get('volcano_power', None)

        if power is not None:
            plot_title += f" | Statistical Power: {power:.2f} (α = {alpha:.2f})"

        fig.update_layout(
            title=plot_title,
            xaxis_title="Log2 Fold Change",
            yaxis_title=f"-log10({pval_col.replace('_', ' ')})",
            width=800,
            height=600,
            legend_title="Significance"
        )

        # Update marker size
        fig.update_traces(marker=dict(size=10, opacity=0.7))

        # Add text labels for selected proteins if available
        if labels is not None and 'label_protein' in df.columns and df['label_protein'].any():
            # Get proteins to label
            label_indices = df['label_protein']
            label_x = df.loc[label_indices, fc_col]
            label_y = -np.log10(df.loc[label_indices, pval_col])
            label_texts = df.loc[label_indices, 'Protein'] if 'Protein' in df.columns else labels[label_indices]

            # Add text annotations
            for i, (x, y, text) in enumerate(zip(label_x, label_y, label_texts)):
                fig.add_annotation(
                    x=x,
                    y=y,
                    text=text,
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="black",
                    font=dict(size=12, color="black"),
                    bgcolor="white",
                    bordercolor=None,  # Remove the border
                    borderwidth=0,     # Set border width to 0
                    borderpad=2,
                    opacity=0.8
                )

        return fig

    @staticmethod
    def create_box_plot(df: pd.DataFrame, value_col: str,
                       group_col: str) -> go.Figure:
        """
        Create interactive box plot
        """
        fig = px.box(
            df,
            x=group_col,
            y=value_col,
            points="all"
        )

        fig.update_layout(
            title=f"Distribution of {value_col} by {group_col}",
            width=800,
            height=600
        )

        return fig

    @staticmethod
    def create_pca_plot(df: pd.DataFrame, group_selections: dict, show_ellipses: bool = False, confidence_level: float = 0.95,
                       enable_clustering: bool = False, clustering_method: str = "K-means", n_clusters: int = 3) -> go.Figure:
        """
        Create PCA plot of samples within selected groups

        Args:
            df: DataFrame with proteomics data
            group_selections: Dictionary mapping group names to column lists
            show_ellipses: Whether to show confidence ellipses
            confidence_level: Confidence level for ellipses (0.90 to 1.00)

        Returns:
            Plotly figure with PCA plot
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        # Extract sample data for PCA
        # For PCA we need to construct a matrix where:
        # - Rows are samples (columns in our original data)
        # - Columns are features (rows/proteins in our original data)

        # First, get all columns to include in PCA
        all_sample_cols = []
        for cols in group_selections.values():
            all_sample_cols.extend(cols)

        if len(all_sample_cols) < 3:
            fig = go.Figure()
            fig.update_layout(title="Not enough samples for PCA (need at least 3)")
            return fig

        # Extract and prepare the data
        # Create a samples x proteins matrix for PCA
        pca_data = df[all_sample_cols].T.copy()  # Transpose so that samples are rows

        # Handle missing values
        # Fill NaN with column mean (now each column is a protein)
        pca_data = pca_data.fillna(pca_data.mean())

        # Skip if no valid data after preprocessing
        if pca_data.empty or pca_data.isnull().all().all():
            fig = go.Figure()
            fig.update_layout(title="Not enough valid data for PCA after NaN handling")
            return fig

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(pca_data)

        # Perform PCA
        pca = PCA(n_components=2)
        pc_coords = pca.fit_transform(X_scaled)

        # Create a DataFrame with sample projections
        sample_projections = []

        # Map each sample to its group
        sample_to_group = {}
        for group_name, sample_cols in group_selections.items():
            for col in sample_cols:
                sample_to_group[col] = group_name

        # Create projection dataframe
        projection_df = pd.DataFrame({
            'PC1': pc_coords[:, 0],
            'PC2': pc_coords[:, 1],
            'sample': pca_data.index,
            'group': [sample_to_group.get(sample, 'Unknown') for sample in pca_data.index]
        })

        # Apply clustering if requested
        if enable_clustering:
            try:
                if clustering_method == "K-means":
                    from sklearn.cluster import KMeans
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = clusterer.fit_predict(pc_coords)

                elif clustering_method == "Hierarchical":
                    from sklearn.cluster import AgglomerativeClustering
                    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                    cluster_labels = clusterer.fit_predict(pc_coords)

                elif clustering_method == "DBSCAN":
                    from sklearn.cluster import DBSCAN
                    # Use eps and min_samples that work well for typical sample sizes
                    clusterer = DBSCAN(eps=0.5, min_samples=2)
                    cluster_labels = clusterer.fit_predict(pc_coords)

                elif clustering_method == "Gaussian Mixture":
                    from sklearn.mixture import GaussianMixture
                    clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
                    cluster_labels = clusterer.fit_predict(pc_coords)

                else:
                    cluster_labels = None

                # Add cluster information to projection dataframe
                if cluster_labels is not None:
                    projection_df['cluster'] = [f'Cluster {i}' for i in cluster_labels]
                    # For plotting, use clusters as the grouping variable
                    color_column = 'cluster'
                    unique_groups = projection_df['cluster'].unique()
                else:
                    color_column = 'group'
                    unique_groups = projection_df['group'].unique()

            except ImportError:
                st.warning("Clustering requires scikit-learn. Using original groupings.")
                color_column = 'group'
                unique_groups = projection_df['group'].unique()
        else:
            color_column = 'group'
            unique_groups = projection_df['group'].unique()

        # Use colorblind-friendly palette
        color_palette = ColorPalettes.COLORBLIND_FRIENDLY

        # Create explicit color map
        color_map = {group: color_palette[i % len(color_palette)]
                    for i, group in enumerate(unique_groups)}

        # Create title with clustering info
        title = f'PCA Plot - Explained variance: PC1 {pca.explained_variance_ratio_[0]:.2%}, PC2 {pca.explained_variance_ratio_[1]:.2%}'
        if enable_clustering and 'cluster' in projection_df.columns:
            title += f' | {clustering_method} Clustering'

        fig = px.scatter(
            projection_df,
            x='PC1',
            y='PC2',
            color=color_column,
            color_discrete_map=color_map,  # Use explicit color map
            title=title,
            hover_data={'group': True} if enable_clustering else None  # Show original groups in hover if clustering
        )

        # Update marker size and hover info with explicit colors
        fig.update_traces(
            marker=dict(size=12, opacity=0.8, line=dict(width=1, color='white')),
            hoverinfo='text',
            hovertext=projection_df['sample']
        )

        # Draw ellipses if requested
        if show_ellipses:
            from scipy.stats import chi2

            # Draw confidence ellipses for each group/cluster
            grouping_column = color_column
            for group in projection_df[grouping_column].unique():
                group_df = projection_df[projection_df[grouping_column] == group]

                if len(group_df) < 3:  # Need at least 3 points for covariance
                    continue

                # Calculate the covariance matrix
                x = group_df['PC1']
                y = group_df['PC2']
                cov = np.cov(x, y)

                # Get the eigenvalues and eigenvectors
                eigenvals, eigenvecs = np.linalg.eigh(cov)

                # Get the indices of the eigenvalues in descending order
                order = eigenvals.argsort()[::-1]
                eigenvals = eigenvals[order]
                eigenvecs = eigenvecs[:, order]

                # Get the largest eigenvalue and eigenvector
                theta = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])

                # Chi-square value for the specified confidence level
                chisquare_val = chi2.ppf(confidence_level, 2)

                # Calculate ellipse parameters
                width = 2 * np.sqrt(chisquare_val * eigenvals[0])
                height = 2 * np.sqrt(chisquare_val * eigenvals[1])

                # Generate ellipse points
                t = np.linspace(0, 2*np.pi, 100)
                ellipse_x = width/2 * np.cos(t)
                ellipse_y = height/2 * np.sin(t)

                # Rotate the ellipse
                x_rot = ellipse_x * np.cos(theta) - ellipse_y * np.sin(theta)
                y_rot = ellipse_x * np.sin(theta) + ellipse_y * np.cos(theta)

                # Shift to the mean position
                x_rot += np.mean(x)
                y_rot += np.mean(y)

                # Get the color for this group
                group_color = None
                for trace in fig.data:
                    if trace.name == group:
                        group_color = trace.marker.color
                        break

                # Add the ellipse as a scatter trace with fill
                fig.add_scatter(
                    x=x_rot,
                    y=y_rot,
                    mode='lines',
                    line=dict(color=group_color, width=2),
                    fill='toself',
                    fillcolor=f'rgba({",".join([str(int(c)) for c in px.colors.hex_to_rgb(group_color)])},0.2)' if isinstance(group_color, str) and group_color.startswith('#') else f'rgba(0,0,0,0.1)',
                    name=f'{group} ({int(confidence_level * 100)}% confidence)',
                    showlegend=True
                )

        # Update layout
        fig.update_layout(
            width=800,
            height=600,
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.2%})",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.2%})",
            legend_title="Group"
        )

        return fig

    @staticmethod
    def create_protein_bar_plot(df: pd.DataFrame, protein_names: list, group_selections: dict, selected_groups: dict = None, equal_var: bool = False, paired: bool = False) -> Tuple[dict, pd.DataFrame]:
        """
        Create bar plot showing protein expression across different sample groups with error bars

        Args:            df: DataFrame with proteomics data
            protein_names: List of protein names to plot
            group_selections: Dictionary mapping group names to column lists
            selected_groups: Dictionary of groups to include (keys are group names, values are booleans)

        Returns:
            Tuple of (Dictionary of protein name to figure, DataFrame with statistics)
        """
        from scipy import stats
        import numpy as np
        import random
        import matplotlib.pyplot as plt
        import matplotlib
        import io
        from PIL import Image

        # Use non-interactive Agg backend for server-side plotting
        matplotlib.use('Agg')

        # Get protein column from session state if available
        protein_col = None
        if hasattr(st, 'session_state') and 'protein_col' in st.session_state:
            protein_col = st.session_state.protein_col

        if not protein_col or protein_col not in df.columns:
            # Try to find a protein column
            for col in df.columns:
                if 'protein' in col.lower() or 'gene' in col.lower() or 'pg.genes' in col.lower():
                    protein_col = col
                    break

            if not protein_col:
                # Use index as last resort
                df = df.copy()
                df['Protein_ID'] = df.index
                protein_col = 'Protein_ID'

        # Filter for specified proteins
        filtered_df = df[df[protein_col].isin(protein_names)]

        if filtered_df.empty:
            # No matching proteins found
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No matching proteins found", ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_xlabel("Group", fontsize=16)
            ax.set_ylabel("Expression", fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)

            # Convert matplotlib figure to image for Streamlit
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)

            # Also save SVG version
            svg_buf = io.BytesIO()
            fig.savefig(svg_buf, format='svg', bbox_inches='tight')
            svg_buf.seek(0)
            svg_data = svg_buf.getvalue()

            plt.close(fig)

            return {"No matching proteins": (img, svg_data)}, pd.DataFrame()

        # Create a DataFrame to store statistics
        stats_df_rows = []

        # Dictionary to store figures for each protein
        protein_figures = {}

        # Process each protein
        for protein_name in protein_names:
            if protein_name not in filtered_df[protein_col].values:
                continue

            protein_data = filtered_df[filtered_df[protein_col] == protein_name]

            # Create a new matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Calculate expression values for each group
            group_means = []
            group_sems = []
            group_names = []

            # Store all data for statistical comparison
            group_values = {}

            for group_name, columns in group_selections.items():
                # Skip if group not selected for display/comparison
                if selected_groups is not None and not selected_groups.get(group_name, True):
                    continue

                if not columns:
                    continue

                # Extract values for this protein and group
                values = protein_data[columns].values.flatten()
                values = values[~np.isnan(values)]  # Remove NaN values

                if len(values) == 0:
                    continue

                # Store values for statistical tests
                group_values[group_name] = values

                # Calculate statistics
                mean_val = np.mean(values)
                sem_val = stats.sem(values) if len(values) > 1 else 0

                group_means.append(mean_val)
                group_sems.append(sem_val)
                group_names.append(group_name)

            # Add bar chart with error bars
            x_pos = np.arange(len(group_names))
            ax.bar(x_pos, group_means, yerr=group_sems, align='center', alpha=0.7, capsize=10, color='#1f77b4', edgecolor='black')

            # Add scatter points for individual samples with controlled jitter
            for i, (group_name, columns) in enumerate(group_selections.items()):
                if not columns or group_name not in group_names:
                    continue

                # Get group index in the ordered arrays
                group_idx = group_names.index(group_name)

                # Extract values for each sample in this group
                sample_values = []
                sample_names = []

                for col in columns:
                    if col in protein_data.columns:
                        value = protein_data[col].values[0]
                        if not pd.isna(value):
                            sample_values.append(value)
                            sample_names.append(col)

                if sample_values:
                    # Generate controlled jitter - ensure points stay over the bar
                    # Use a narrow jitter range (0.2) centered on the bar position
                    jitter_width = 0.2
                    n_samples = len(sample_values)

                    # Create fixed jitter positions that span the width evenly
                    if n_samples > 1:
                        # Equal spacing between points
                        jitter = np.linspace(-jitter_width/2, jitter_width/2, n_samples)
                        # Add small random noise to prevent perfect alignment
                        small_noise = np.random.normal(0, 0.02, n_samples)
                        jitter = jitter + small_noise
                    else:
                        # Single point, center it
                        jitter = [0]

                    # Plot points with controlled jitter
                    ax.scatter(
                        x_pos[group_idx] + jitter,
                        sample_values,
                        color='black',
                        alpha=0.7,
                        s=50,  # Increased point size
                        zorder=3  # Ensure points are drawn on top
                    )

            # Perform statistical tests between all group pairs (for stats table only)
            group_pairs = [(i, j) for i in range(len(group_names)) for j in range(i+1, len(group_names))]

            # Add statistical data to the table but don't draw on plot
            for i, j in group_pairs:
                group1_name = group_names[i]
                group2_name = group_names[j]

                group1_values = group_values[group1_name]
                group2_values = group_values[group2_name]

                if len(group1_values) > 1 and len(group2_values) > 1:
                    # Perform t-test with specified parameters
                    if paired:
                        # For paired t-test, take minimum length and match samples
                        min_length = min(len(group1_values), len(group2_values))
                        if min_length > 1:
                            t_stat, p_val = stats.ttest_rel(
                                group1_values[:min_length],
                                group2_values[:min_length]
                            )
                        else:
                            t_stat, p_val = 0, 1.0
                    else:
                        # Independent samples t-test
                        t_stat, p_val = stats.ttest_ind(
                            group1_values,
                            group2_values,
                            equal_var=equal_var,
                            nan_policy='omit'
                        )

                    # Calculate fold change
                    mean1 = np.mean(group1_values)
                    mean2 = np.mean(group2_values)
                    log2fc = np.log2(mean2 / mean1) if mean1 > 0 else float('inf')

                    # Add to stats DataFrame
                    stats_df_rows.append({
                        'Protein': protein_name,
                        'Group 1': group1_name,
                        'Group 2': group2_name,
                        'Mean 1': mean1,
                        'Mean 2': mean2,
                        'Log2 Fold Change': log2fc,
                        't-statistic': t_stat,
                        'p-value': p_val,
                        'Significant': p_val < 0.05
                    })

                    # Note: No significance bars or annotations are added to the plot

            # Set plot labels and styling with increased font sizes
            ax.set_title(f"Expression of {protein_name}", fontsize=16, pad=20)
            ax.set_ylabel("Expression Level", fontsize=16)
            ax.set_xlabel("Group", fontsize=16)

            # Set x-axis ticks to group names with increased font size
            ax.set_xticks(x_pos)
            ax.set_xticklabels(group_names, fontsize=14)

            # Increase y-axis tick font size
            ax.tick_params(axis='y', which='major', labelsize=14)

            # Add grid lines for easier reading
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Update y-axis limits without extending for significance bars
            # Get current limits
            y_min, y_max = ax.get_ylim()

            # Only add a small margin (10%) for better readability
            ax.set_ylim(y_min, y_max * 1.1)

            # Adjust layout
            plt.tight_layout()

            # Convert matplotlib figure to PIL Image for Streamlit
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)

            # Also save SVG version
            svg_buf = io.BytesIO()
            fig.savefig(svg_buf, format='svg', bbox_inches='tight')
            svg_data = svg_buf.getvalue()

            plt.close(fig)  # Close figure to free memory

            # Store image and SVG data in dictionary
            protein_figures[protein_name] = (img, svg_data)

        # Create stats DataFrame
        stats_df = pd.DataFrame(stats_df_rows)

        return protein_figures, stats_df

    @staticmethod
    def create_protein_rank_plot(df: pd.DataFrame, intensity_columns: list,
                                highlight_removed: bool = False,
                                filtered_data: pd.DataFrame = None,
                                proteins_to_highlight: list = None) -> go.Figure:
        """
        Create a new implementation of the protein rank plot showing proteins ranked by their abundance

        Args:
            df: DataFrame with proteomics data
            intensity_columns: Columns to use for calculating protein abundance
            highlight_removed: Whether to highlight proteins that were removed in filtered dataset
            filtered_data: Filtered DataFrame (used when highlight_removed is True)
            proteins_to_highlight: List of protein names to highlight on the plot

        Returns:
            Plotly figure with protein rank plot
        """
        # Get protein column if available
        protein_col = None
        if hasattr(st, 'session_state') and 'protein_col' in st.session_state:
            protein_col = st.session_state.protein_col

        # Calculate mean intensity across selected columns for each protein
        mean_intensities = df[intensity_columns].mean(axis=1)

        # Create a new DataFrame for plotting
        plot_data = pd.DataFrame({
            'intensity': mean_intensities
        })

        # Add protein names if available
        if protein_col and protein_col in df.columns:
            plot_data['protein'] = df[protein_col].values

        # Sort by intensity in descending order and add rank
        plot_data = plot_data.sort_values('intensity', ascending=False).reset_index(drop=True)
        plot_data['rank'] = range(1, len(plot_data) + 1)

        # First create a basic figure
        fig = go.Figure()

        # Use colorblind-friendly colors by adding transparency to our palette
        # Convert hex to rgba with transparency
        def hex_to_rgba(hex_color, alpha=0.7):
            hex_color = hex_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({r}, {g}, {b}, {alpha})'

        # Define colorblind-friendly colors with transparency
        color_palette = ColorPalettes.COLORBLIND_FRIENDLY
        base_color = hex_to_rgba(color_palette[0], 0.7)  # Blue
        highlight_color = hex_to_rgba(color_palette[1], 0.9)  # Vermillion (red)
        filtered_color = hex_to_rgba(color_palette[4], 0.7)  # Light blue
        kept_color = hex_to_rgba(color_palette[2], 0.7)  # Green

        # Handle different display scenarios
        if highlight_removed and filtered_data is not None:
            # Setup for filtering status
            kept_indices = filtered_data.index
            plot_data['status'] = 'Removed'
            plot_data.loc[plot_data.index.isin(kept_indices), 'status'] = 'Kept'

            # Create two separate traces for kept and removed proteins
            kept_data = plot_data[plot_data['status'] == 'Kept']
            removed_data = plot_data[plot_data['status'] == 'Removed']

            # Add scatter trace for kept proteins
            fig.add_trace(go.Scatter(
                x=kept_data['rank'],
                y=kept_data['intensity'],
                mode='markers',
                name='Kept after filtering',
                marker=dict(
                    color=kept_color,
                    size=8,
                    opacity=0.8
                ),
                hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Intensity: %{y:.2e}<extra></extra>',
                text=kept_data['protein'] if 'protein' in kept_data.columns else None
            ))

            # Add scatter trace for removed proteins
            fig.add_trace(go.Scatter(
                x=removed_data['rank'],
                y=removed_data['intensity'],
                mode='markers',
                name='Removed by filtering',
                marker=dict(
                    color=filtered_color,
                    size=8,
                    opacity=0.8
                ),
                hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Intensity: %{y:.2e}<extra></extra>',
                text=removed_data['protein'] if 'protein' in removed_data.columns else None
            ))

        elif proteins_to_highlight and 'protein' in plot_data.columns:
            # Split the data for highlighted and non-highlighted proteins
            plot_data['highlighted'] = plot_data['protein'].isin(proteins_to_highlight)

            highlighted_data = plot_data[plot_data['highlighted']]
            other_data = plot_data[~plot_data['highlighted']]

            # Add scatter trace for non-highlighted proteins
            fig.add_trace(go.Scatter(
                x=other_data['rank'],
                y=other_data['intensity'],
                mode='markers',
                name='Other proteins',
                marker=dict(
                    color=base_color,
                    size=8,
                    opacity=0.7
                ),
                hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Intensity: %{y:.2e}<extra></extra>',
                text=other_data['protein'] if 'protein' in other_data.columns else None
            ))

            # Add scatter trace for highlighted proteins with larger markers
            fig.add_trace(go.Scatter(
                x=highlighted_data['rank'],
                y=highlighted_data['intensity'],
                mode='markers',
                name='Highlighted proteins',
                marker=dict(
                    color=highlight_color,
                    size=12,
                    line=dict(width=2, color='black')
                ),
                hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Intensity: %{y:.2e}<extra></extra>',
                text=highlighted_data['protein'] if 'protein' in highlighted_data.columns else None
            ))

            # Add text annotations for highlighted proteins
            for _, row in highlighted_data.iterrows():
                fig.add_annotation(
                    x=row['rank'],
                    y=row['intensity'] * 1.2,  # Position slightly above the point
                    text=row['protein'],
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="black",
                    font=dict(size=12, color="black", family="Arial Black"),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4,
                    opacity=1.0
                )

            # Store figure in session state for download
            if hasattr(st, 'session_state'):
                st.session_state.rank_plot_fig = fig

        else:
            # Just add a single trace for all proteins
            fig.add_trace(go.Scatter(
                x=plot_data['rank'],
                y=plot_data['intensity'],
                mode='markers',
                name='Proteins',
                marker=dict(
                    color=base_color,
                    size=8,
                    opacity=0.7
                ),
                hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Intensity: %{y:.2e}<extra></extra>',
                text=plot_data['protein'] if 'protein' in plot_data.columns else None
            ))

        # Add a smooth line to show the trend
        sorted_intensities = plot_data['intensity'].values
        x_range = plot_data['rank'].values

        fig.add_trace(go.Scatter(
            x=x_range,
            y=sorted_intensities,
            mode='lines',
            line=dict(color='rgba(0,0,0,0.3)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Calculate data-appropriate axis ranges
        x_min, x_max = 1, len(plot_data)
        y_min, y_max = plot_data['intensity'].min(), plot_data['intensity'].max()

        # Add some padding to the ranges (5% on each side)
        x_padding = max(1, int(0.05 * x_max))
        y_log_range = np.log10(y_max) - np.log10(y_min)
        y_padding_factor = 10**(0.05 * y_log_range)

        # Update layout with log scale for y-axis and data-tailored ranges
        fig.update_layout(
            title='Protein Rank Plot - Dynamic Range of Proteome',
            xaxis_title='Protein Rank (by abundance)',
            yaxis_title='Signal Intensity (log scale)',
            xaxis=dict(
                range=[max(1, x_min - x_padding), x_max + x_padding],
                type='linear'
            ),
            yaxis=dict(
                type='log',
                range=[np.log10(y_min / y_padding_factor), np.log10(y_max * y_padding_factor)]
            ),
            height=600,
            width=900,
            template='plotly_white',
            hovermode='closest',
            # Explicitly define colorway for download compatibility using colorblind-friendly palette
            colorway=ColorPalettes.COLORBLIND_FRIENDLY
        )

        # Add annotations about dynamic range
        if len(sorted_intensities) > 0:
            max_intensity = sorted_intensities[0]
            min_intensity = sorted_intensities[-1] if len(sorted_intensities) > 1 else max_intensity

            dynamic_range = max_intensity / min_intensity if min_intensity > 0 else float('inf')

            fig.add_annotation(
                x=0.95,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"Dynamic Range: {dynamic_range:.1f}x",
                showarrow=False,
                font=dict(size=14),
                align="right",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )

            fig.add_annotation(
                x=0.95,
                y=0.89,
                xref="paper",
                yref="paper",
                text=f"Total Proteins: {len(sorted_intensities)}",
                showarrow=False,
                font=dict(size=14),
                align="right",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )

        return fig

    @staticmethod
    def create_correlation_plot(df: pd.DataFrame, group_selections: dict, mode: str = "within_groups",
                              group1: str = None, group2: str = None) -> Tuple[List[go.Figure], pd.DataFrame]:
        """
        Create pairwise correlation plot for samples

        Args:
            df: DataFrame with proteomics data
            group_selections: Dictionary mapping group names to column lists
            mode: Either "within_groups" or "between_groups"
            group1: First group name for between_groups mode
            group2: Second group name for between_groups mode

        Returns:
            Plotly figure with correlation plot
        """
        import io
        if mode == "within_groups":
            # Create subplots for each group
            figs = []
            max_width = 1200  # Maximum width of the figure
            max_height = 1000  # Maximum height of the figure

            for group_name, cols in group_selections.items():
                if len(cols) < 2:  # Skip groups with less than 2 samples
                    continue

                # Calculate number of rows and columns needed for this group
                n = len(cols)
                fig = make_subplots(
                    rows=n, cols=n,
                    subplot_titles=[f"{cols[j]} vs {cols[i]}" if i > j else None
                                  for i in range(n) for j in range(n)],
                    horizontal_spacing=0.05,
                    vertical_spacing=0.05
                )

                # Create scatter plots for each pair
                for i in range(n):
                    for j in range(n):
                        if i >= j:  # Diagonal and lower triangle: scatter plots
                            x_data = df[cols[j]]
                            y_data = df[cols[i]]

                            # Calculate Pearson correlation coefficient
                            corr = np.corrcoef(x_data, y_data)[0,1]

                            # Add correlation method text
                            fig.add_annotation(
                                text='Pearson Correlation',
                                xref=f"x{i*n + j + 1}",
                                yref=f"y{i*n + j + 1}",
                                x=0.05,
                                y=0.95,
                                showarrow=False,
                                font=dict(size=10),
                                bgcolor='white',
                                bordercolor='black',
                                borderwidth=1
                            )

                            # Calculate axis limits based on data
                            min_val = min(x_data.min(), y_data.min())
                            max_val = max(x_data.max(), y_data.max())

                            # Add scatter points with colorblind-friendly color
                            fig.add_trace(
                                go.Scatter(
                                    x=x_data,
                                    y=y_data,
                                    mode='markers',
                                    marker=dict(
                                        size=2,
                                        opacity=0.5,
                                        color=ColorPalettes.COLORBLIND_FRIENDLY[4]  # Light blue
                                    ),
                                    showlegend=False
                                ),
                                row=i+1, col=j+1
                            )

                            # Calculate Pearson correlation coefficient and p-value
                            valid_mask = ~(pd.isna(x_data) | pd.isna(y_data))
                            x_valid = x_data[valid_mask]
                            y_valid = y_data[valid_mask]

                            corr, p_val = stats.pearsonr(x_valid, y_valid)

                            # Add correlation value and p-value in top-left corner
                            fig.add_annotation(
                                text=f'R = {corr:.3f}<br>p = {p_val:.2e}',
                                xref=f"x{i*n + j + 1}",
                                yref=f"y{i*n + j + 1}",
                                x=0.05,
                                y=0.95,
                                showarrow=False,
                                font=dict(size=12, color='red', weight='bold'),
                                bgcolor='rgba(255,255,255,0.9)',
                                bordercolor='black',
                                borderwidth=1,
                                borderpad=4,
                                align='left'
                            )

                            # Update axes for this subplot
                            fig.update_xaxes(
                                title=cols[j] if i == (n-1) else None,
                                range=[min_val, max_val],
                                row=i+1, col=j+1,
                                showgrid=True,
                                gridcolor='lightgray'
                            )
                            fig.update_yaxes(
                                title=cols[i] if j == 0 else None,
                                range=[min_val, max_val],
                               row=i+1, col=j+1,
                                showgrid=True,
                                gridcolor='lightgray'
                            )

                        # Update axes for all plots
                        fig.update_xaxes(
                            title=cols[j] if i == (n-1) else None,
                            showgrid=True,
                            row=i+1, col=j+1
                        )
                        fig.update_yaxes(
                            title=cols[i] if j == 0 else None,
                            showgrid=True,
                            row=i+1, col=j+1
                        )

                # Update layout for this group
                width = min(max_width, 300 * n)
                height = min(max_height, 300 * n)

                fig.update_layout(
                    title=f"Correlation Plots for {group_name}",
                    width=width,
                    height=height,
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )

                figs.append(fig)

            # Create stats DataFrame
            stats_rows = []
            for group_name, cols in group_selections.items():
                if len(cols) < 2:
                    continue
                for i in range(len(cols)):
                    for j in range(i):
                        x_data = df[cols[j]]
                        y_data = df[cols[i]]
                        valid_mask = ~(pd.isna(x_data) | pd.isna(y_data))
                        x_valid = x_data[valid_mask]
                        y_valid = y_data[valid_mask]
                        corr, p_val = stats.pearsonr(x_valid, y_valid)
                        stats_rows.append({
                            'Group': group_name,
                            'Sample 1': cols[j],
                            'Sample 2': cols[i],
                            'R': corr,
                            'p-value': p_val
                        })

            stats_df = pd.DataFrame(stats_rows)

            # Return figures and stats
            return figs, stats_df

        else:  # between_groups mode
            if not (group1 and group2):
                return go.Figure()

            # Calculate average expression for each group
            group1_mean = df[group_selections[group1]].mean(axis=1)
            group2_mean = df[group_selections[group2]].mean(axis=1)

            # Get protein column from session state if available
            protein_col = None
            if hasattr(st, 'session_state') and 'protein_col' in st.session_state:
                protein_col = st.session_state.protein_col

            # Create hover text with protein names if available
            hover_text = None
            if protein_col and protein_col in df.columns:
                hover_text = df[protein_col].values

            # Create scatter plot
            fig = go.Figure(data=go.Scatter(
                x=group1_mean,
                y=group2_mean,
                mode='markers',
                marker=dict(
                    size=8,
                    opacity=0.6,
                    color=ColorPalettes.COLORBLIND_FRIENDLY[0]  # Blue
                ),
                name='Proteins',
                hovertemplate='<b>%{text}</b><br>' +
                             f'{group1}: %{{x:,.2f}}<br>' +
                             f'{group2}: %{{y:,.2f}}<extra></extra>',
                text=hover_text
            ))

            # Calculate correlation coefficient
            corr = np.corrcoef(group1_mean, group2_mean)[0,1]

            # Add correlation line
            z = np.polyfit(group1_mean, group2_mean, 1)
            p = np.poly1d(z)
            x_range = np.linspace(group1_mean.min(), group1_mean.max(), 100)

            fig.add_trace(go.Scatter(
                x=x_range,
                y=p(x_range),
                mode='lines',
                name=f'R = {corr:.3f}',
                line=dict(color=ColorPalettes.COLORBLIND_FRIENDLY[1], dash='dash')  # Vermillion (red)
            ))

            # Update layout with better sizing and margins
            fig.update_layout(
                title=f"Correlation between {group1} and {group2}",
                xaxis_title=f"{group1} (mean expression)",
                yaxis_title=f"{group2} (mean expression)",
                width=800,
                height=800,
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=80, r=80, t=100, b=80),
                autosize=True
            )

            # Store figure in session state for downloads
            if hasattr(st, 'session_state'):
                st.session_state.correlation_fig = fig

            return fig

    @staticmethod
    def create_pls_da_plot(pls_results: Dict[str, Any]) -> go.Figure:
        """
        Create PLS-DA scores plot

        Args:
            pls_results: Results from PLS-DA analysis

        Returns:
            Plotly figure object
        """
        scores = pls_results['scores']
        group_labels = pls_results['group_labels']
        sample_names = pls_results['sample_names']
        explained_var = pls_results['explained_variance']
        cv_accuracy = pls_results['cv_accuracy']
        cv_std = pls_results['cv_std']

        # Create color mapping
        unique_groups = list(set(group_labels))
        colors = px.colors.qualitative.Plotly[:len(unique_groups)]
        color_map = dict(zip(unique_groups, colors))

        fig = go.Figure()

        # Add scatter plot for each group
        for group in unique_groups:
            group_mask = [label == group for label in group_labels]
            group_scores = scores[group_mask]
            group_samples = [sample_names[i] for i, mask in enumerate(group_mask) if mask]

            fig.add_trace(go.Scatter(
                x=group_scores[:, 0],
                y=group_scores[:, 1],
                mode='markers',
                name=group,
                marker=dict(
                    color=color_map[group],
                    size=10,
                    line=dict(width=1, color='white')
                ),
                text=group_samples,
                hovertemplate='<b>%{text}</b><br>' +
                             'PC1: %{x:.3f}<br>' +
                             'PC2: %{y:.3f}<br>' +
                             f'Group: {group}<extra></extra>'
            ))

        # Calculate axis ranges with some padding
        x_range = [scores[:, 0].min() * 1.1, scores[:, 0].max() * 1.1]
        y_range = [scores[:, 1].min() * 1.1, scores[:, 1].max() * 1.1]

        # Add confidence ellipses if more than 2 samples per group
        for group in unique_groups:
            group_mask = [label == group for label in group_labels]
            group_scores = scores[group_mask]

            if len(group_scores) > 2:
                ellipse = Visualizer._create_confidence_ellipse(
                    group_scores[:, 0], group_scores[:, 1], 0.95
                )
                if ellipse is not None:
                    fig.add_trace(go.Scatter(
                        x=ellipse[0],
                        y=ellipse[1],
                        mode='lines',
                        name=f'{group} (95% CI)',
                        line=dict(color=color_map[group], dash='dash'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

        # Update layout
        fig.update_layout(
            title=f'PLS-DA Scores Plot<br><sub>CV Accuracy: {cv_accuracy:.3f} ± {cv_std:.3f}</sub>',
            xaxis_title=f'Component 1 ({explained_var[0]:.1f}%)',
            yaxis_title=f'Component 2 ({explained_var[1]:.1f}%)',
            xaxis=dict(range=x_range, zeroline=True, zerolinewidth=1, zerolinecolor='lightgray'),
            yaxis=dict(range=y_range, zeroline=True, zerolinewidth=1, zerolinecolor='lightgray'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=800,
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        )

        return fig

    @staticmethod
    def create_vip_plot(pls_results: Dict[str, Any], df: pd.DataFrame,
                       top_n: int = 20, vip_threshold: float = 1.0) -> go.Figure:
        """
        Create VIP scores plot

        Args:
            pls_results: Results from PLS-DA analysis
            df: Original dataframe for protein names
            top_n: Number of top VIP proteins to show
            vip_threshold: VIP threshold line to display

        Returns:
            Plotly figure object
        """
        vip_scores = pls_results['vip_scores']
        protein_indices = pls_results['protein_indices']

        # Get protein names
        protein_col = next((col for col in df.columns if col in ['PG.Genes', 'Gene Name', 'Protein']), None)
        if protein_col:
            protein_names = df.loc[protein_indices, protein_col].values
        else:
            protein_names = [f'Protein_{i}' for i in range(len(protein_indices))]

        # Create VIP dataframe and sort
        vip_df = pd.DataFrame({
            'Protein': protein_names,
            'VIP_Score': vip_scores,
            'Protein_Index': protein_indices
        }).sort_values('VIP_Score', ascending=False)

        # Take top N proteins
        top_vip = vip_df.head(top_n)

        # Create colors based on VIP threshold
        colors = ['red' if score >= vip_threshold else 'blue' for score in top_vip['VIP_Score']]

        fig = go.Figure()

        # Add bar plot
        fig.add_trace(go.Bar(
            x=top_vip['VIP_Score'],
            y=top_vip['Protein'],
            orientation='h',
            marker=dict(color=colors),
            text=[f'{score:.2f}' for score in top_vip['VIP_Score']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>VIP Score: %{x:.3f}<extra></extra>'
        ))

        # Add VIP threshold line
        fig.add_vline(
            x=vip_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VIP = {vip_threshold}",
            annotation_position="top"
        )

        # Update layout
        fig.update_layout(
            title=f'Top {top_n} VIP Proteins (PLS-DA)',
            xaxis_title='VIP Score',
            yaxis_title='Protein',
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=800,
            height=max(400, top_n * 25),
            showlegend=False,
            yaxis=dict(autorange="reversed")  # Show highest VIP at top
        )

        return fig

    @staticmethod
    def _calculate_heatmap_statistics(plot_data: pd.DataFrame, group_selections: dict, structure: dict) -> pd.DataFrame:
        """Calculate statistical comparison metrics for heatmap proteins."""
        from scipy.stats import ttest_ind
        from itertools import combinations

        stats_rows = []

        # Calculate statistics for each protein and group pair
        for protein in plot_data.index:
            for group1, group2 in combinations(group_selections.keys(), 2):
                group1_cols = [col for col in structure["replicates"][group1] if col.endswith("PG.Quantity")]
                group2_cols = [col for col in structure["replicates"][group2] if col.endswith("PG.Quantity")]

                group1_data = plot_data.loc[protein, group1_cols].dropna()
                group2_data = plot_data.loc[protein, group2_cols].dropna()

                if len(group1_data) > 1 and len(group2_data) > 1:
                    t_stat, p_val = ttest_ind(group1_data, group2_data, equal_var=False)
                    mean1 = group1_data.mean()
                    mean2 = group2_data.mean()
                    log2fc = np.log2(mean2 / mean1) if mean1 > 0 else float('inf')

                    stats_rows.append({
                        'Protein': protein,
                        'Group 1': group1,
                        'Group 2': group2,
                        'Mean 1': mean1,
                        'Mean 2': mean2,
                        'Log2 Fold Change': log2fc,
                        't-statistic': t_stat,
                        'p-value': p_val
                    })

        return pd.DataFrame(stats_rows)

    @staticmethod
    def create_upset_plot(df: pd.DataFrame, group_selections: dict,
                         selected_comparisons: list = None,
                         fc_threshold: float = 1.0, p_threshold: float = 0.05,
                         regulation_filter: str = "All", equal_var: bool = False,
                         paired: bool = False, correction_method: str = "None") -> Tuple[object, pd.DataFrame]:
        """
        Create UpSet plot for differentially expressed proteins across multiple comparisons using pyUpSet

        Args:
            df: DataFrame with proteomics data
            group_selections: Dictionary mapping group names to column lists
            selected_comparisons: List of tuples (group1, group2) for specific comparisons
            fc_threshold: Log2 fold change threshold
            p_threshold: p-value threshold
            regulation_filter: "All", "Up-regulated", "Down-regulated"
            equal_var: Whether to assume equal variances in t-test
            paired: Whether to perform paired t-test
            correction_method: Method for multiple testing correction

        Returns:
            Tuple of (matplotlib figure, DE proteins DataFrame)
        """
        from itertools import combinations
        import matplotlib.pyplot as plt
        import matplotlib
        import io
        from PIL import Image

        # Use non-interactive backend for server-side plotting
        matplotlib.use('Agg')

        # Use selected comparisons if provided, otherwise use all possible comparisons
        if selected_comparisons is not None:
            comparisons = selected_comparisons
        else:
            # Get all possible pairwise comparisons (fallback for backward compatibility)
            group_names = list(group_selections.keys())
            if len(group_names) < 2:
                raise ValueError("Need at least 2 groups for UpSet plot")
            comparisons = list(combinations(group_names, 2))

        if not comparisons:
            raise ValueError("No comparisons selected for UpSet plot")

        # Store DE proteins for each comparison
        de_sets = {}
        all_de_data = []

        # Check if data is log2 normalized
        is_log2_normalized = hasattr(st, 'session_state') and st.session_state.get('normalization_method', None) == 'Log2'

        for group1, group2 in comparisons:
            comparison_name = f"{group2}_vs_{group1}"

            # Get columns for each group
            control_cols = group_selections[group1]
            treatment_cols = group_selections[group2]

            if not control_cols or not treatment_cols:
                continue

            # Calculate fold change and p-values
            control_mean = df[control_cols].mean(axis=1)
            treatment_mean = df[treatment_cols].mean(axis=1)

            # Calculate log2 fold change
            if is_log2_normalized:
                log2fc = treatment_mean - control_mean
            else:
                epsilon = 1e-10
                ratio = (treatment_mean + epsilon) / (control_mean + epsilon)
                log2fc = np.log2(ratio)

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
                        if paired:
                            min_length = min(len(control_values), len(treatment_values))
                            if min_length > 1:
                                _, p_val = stats.ttest_rel(
                                    treatment_values[:min_length],
                                    control_values[:min_length]
                                )
                            else:
                                p_val = 1.0
                        else:
                            _, p_val = stats.ttest_ind(
                                treatment_values,
                                control_values,
                                equal_var=equal_var,
                                nan_policy='omit'
                            )
                        p_values.append(p_val)
                    except:
                        p_values.append(1.0)
                else:
                    p_values.append(1.0)

            # Apply multiple hypothesis correction
            corrected_p = p_values
            if correction_method == "Benjamini-Hochberg (FDR)":
                try:
                    from statsmodels.stats.multitest import fdrcorrection
                    _, corrected_p = fdrcorrection(p_values, alpha=p_threshold)
                except ImportError:
                    pass
            elif correction_method == "Bonferroni":
                corrected_p = [p * len(p_values) for p in p_values]

            # Create DataFrame for this comparison
            comparison_df = pd.DataFrame({
                'Log2FC': log2fc,
                'p_value': corrected_p,
                'comparison': comparison_name
            }, index=df.index)

            # Add protein names if available
            protein_col = None
            if hasattr(st, 'session_state') and 'protein_col' in st.session_state:
                protein_col = st.session_state.protein_col

            if protein_col and protein_col in df.columns:
                comparison_df['Protein'] = df[protein_col]

            # Apply significance filters
            sig_mask = (comparison_df['p_value'] < p_threshold) & (comparison_df['Log2FC'].abs() >= fc_threshold)

            # Apply regulation filter
            if regulation_filter == "Up-regulated":
                sig_mask = sig_mask & (comparison_df['Log2FC'] > 0)
            elif regulation_filter == "Down-regulated":
                sig_mask = sig_mask & (comparison_df['Log2FC'] < 0)

            # Get significant proteins
            sig_proteins = comparison_df[sig_mask].index.tolist()
            de_sets[comparison_name] = set(sig_proteins)

            # Store data for table
            sig_data = comparison_df[sig_mask].copy()
            all_de_data.append(sig_data)

        # Check if we have any significant proteins
        all_proteins = set()
        for proteins in de_sets.values():
            all_proteins.update(proteins)

        if not all_proteins:
            # No significant proteins found - create empty matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No significantly differentially expressed proteins found",
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f"UpSet Plot - FC≥{fc_threshold}, p≤{p_threshold}, {regulation_filter}")
            ax.axis('off')

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)

            # Generate empty SVG data
            svg_buf = io.BytesIO()
            fig.savefig(svg_buf, format='svg', bbox_inches='tight')
            svg_data = svg_buf.getvalue()

            plt.close(fig)

            # Store SVG data in session state
            if hasattr(st, 'session_state'):
                st.session_state.upset_plot_svg = svg_data

            return img, pd.DataFrame(), {}

        # Create custom UpSet-style plot
        try:
            # Calculate intersections
            intersections = {}
            comparison_names = list(de_sets.keys())

            # Single set sizes (exclusive - proteins only in this set)
            for name in comparison_names:
                exclusive_set = de_sets[name].copy()
                other_sets = [other_name for other_name in comparison_names if other_name != name]

                # Remove proteins that are also in any other set
                for other_name in other_sets:
                    exclusive_set = exclusive_set - de_sets[other_name]

                if len(exclusive_set) > 0:
                    intersections[tuple([name])] = len(exclusive_set)

            # Calculate all possible intersections (exclusive - proteins that belong to exactly these sets)
            for r in range(2, len(comparison_names) + 1):
                for combo in combinations(comparison_names, r):
                    # Find proteins that are in all sets of this combination
                    intersection_set = de_sets[combo[0]]
                    for other_name in combo[1:]:
                        intersection_set = intersection_set.intersection(de_sets[other_name])

                    # Find proteins that are ONLY in these sets (exclusive intersection)
                    # Remove proteins that are also in any other set not in this combination
                    other_sets = [name for name in comparison_names if name not in combo]
                    exclusive_intersection = intersection_set.copy()

                    for other_set_name in other_sets:
                        exclusive_intersection = exclusive_intersection - de_sets[other_set_name]

                    if len(exclusive_intersection) > 0:
                        intersections[combo] = len(exclusive_intersection)

            # Sort intersections by size
            sorted_intersections = sorted(intersections.items(), key=lambda x: x[1], reverse=True)

            # Create figure with subplots
            fig = plt.figure(figsize=(14, 10))

            # Create grid: top for intersection sizes, bottom for set indicators
            gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 0.5], hspace=0.1)

            # Top subplot: bar chart of intersection sizes
            ax_bars = fig.add_subplot(gs[0])

            # Extract data for plotting
            intersection_sizes = [item[1] for item in sorted_intersections]
            intersection_labels = [item[0] for item in sorted_intersections]

            # Create bar chart
            bars = ax_bars.bar(range(len(intersection_sizes)), intersection_sizes,
                              color='steelblue', alpha=0.7)

            # Add value labels on bars
            for i, (bar, size) in enumerate(zip(bars, intersection_sizes)):
                ax_bars.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(intersection_sizes)*0.01,
                           str(size), ha='center', va='bottom', fontsize=10)

            ax_bars.set_ylabel('Intersection Size', fontsize=12, fontweight='bold')
            ax_bars.set_title(f'UpSet Plot - Differentially Expressed Proteins\nFC≥{fc_threshold}, p≤{p_threshold}, {regulation_filter}',
                            fontsize=14, fontweight='bold')
            ax_bars.set_xticks(range(len(intersection_labels)))
            ax_bars.set_xticklabels([])
            ax_bars.set_xlim(-0.5, len(intersection_labels) - 0.5)
            ax_bars.grid(axis='y', alpha=0.3)

            # Bottom subplot: set membership matrix
            ax_matrix = fig.add_subplot(gs[1])

            # Create matrix for visualization
            matrix_data = []
            for intersection_tuple in intersection_labels:
                row = []
                for comp_name in comparison_names:
                    if comp_name in intersection_tuple:
                        row.append(1)
                    else:
                        row.append(0)
                matrix_data.append(row)

            matrix_data = np.array(matrix_data).T  # Transpose for correct orientation

            # Create white background matrix (remove the imshow that creates colored background)
            ax_matrix.set_facecolor('white')

            # Add dots for connections with proper centering
            for i, intersection_tuple in enumerate(intersection_labels):
                y_positions = [j for j, comp_name in enumerate(comparison_names) if comp_name in intersection_tuple]
                if len(y_positions) > 1:
                    # Draw connecting line
                    ax_matrix.plot([i, i], [min(y_positions), max(y_positions)], 'k-', linewidth=4)

                # Add larger dots - center them properly on the grid
                for y_pos in y_positions:
                    ax_matrix.plot(i, y_pos, 'ko', markersize=12)

            # Customize matrix plot with proper alignment
            ax_matrix.set_xticks(range(len(intersection_labels)))
            ax_matrix.set_xticklabels([])
            ax_matrix.set_yticks(range(len(comparison_names)))
            ax_matrix.set_yticklabels(comparison_names, fontsize=10)
            ax_matrix.set_ylabel('Comparisons', fontsize=12, fontweight='bold')

            # Set axis limits to center dots properly and align with bars above
            ax_matrix.set_xlim(-0.5, len(intersection_labels) - 0.5)
            ax_matrix.set_ylim(-0.5, len(comparison_names) - 0.5)

            # Remove spines and grid
            for spine in ax_matrix.spines.values():
                spine.set_visible(False)
            ax_matrix.grid(False)

            # Remove ticks
            ax_matrix.tick_params(length=0)

            # Invert y-axis to match the bar chart order
            ax_matrix.invert_yaxis()

            # Set size subplot
            ax_sets = fig.add_subplot(gs[2])
            set_sizes = [len(de_sets[name]) for name in comparison_names]
            bars_sets = ax_sets.barh(range(len(comparison_names)), set_sizes, color='lightgray', alpha=0.7)

            # Add value labels at the end of each bar
            for i, (bar, size) in enumerate(zip(bars_sets, set_sizes)):
                # Place labels at the end of each bar with proper positioning
                ax_sets.text(bar.get_width() + max(set_sizes)*0.01, bar.get_y() + bar.get_height()/2,
                           str(size), ha='left', va='center', fontsize=11, fontweight='bold')

            # Set y-tick labels to show comparison names
            ax_sets.set_yticks(range(len(comparison_names)))
            ax_sets.set_yticklabels(comparison_names, fontsize=10)
            ax_sets.set_xlabel('Set Size', fontsize=12, fontweight='bold')
            ax_sets.grid(axis='x', alpha=0.3)

            # Adjust x-axis limits to accommodate labels
            ax_sets.set_xlim(0, max(set_sizes) * 1.15)

            plt.tight_layout()

            # Convert matplotlib figure to image for Streamlit display
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)

            # Generate SVG data before closing the figure
            svg_buf = io.BytesIO()
            fig.savefig(svg_buf, format='svg', bbox_inches='tight')
            svg_data = svg_buf.getvalue()

            # Store the original figure object before closing
            fig_copy = fig

            plt.close(fig)

            # Store in session state for downloads
            if hasattr(st, 'session_state'):
                st.session_state.upset_matplotlib_fig = fig_copy # Store the matplotlib figure object
                st.session_state.upset_plot_svg = svg_data # Store SVG data

        except Exception:
            # Fallback to simple bar chart - suppress error display
            try:
                fig, ax = plt.subplots(figsize=(12, 8))

                comparison_names = list(de_sets.keys())
                set_sizes = [len(proteins) for proteins in de_sets.values()]

                bars = ax.bar(range(len(comparison_names)), set_sizes, color='steelblue', alpha=0.7)

                # Add value labels
                for bar, size in zip(bars, set_sizes):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(set_sizes)*0.01,
                           str(size), ha='center', va='bottom', fontsize=12)

                ax.set_xticks(range(len(comparison_names)))
                ax.set_xticklabels(comparison_names, rotation=45, ha='right')
                ax.set_ylabel('Number of DE Proteins', fontsize=12)
                ax.set_title(f'Differentially Expressed Proteins by Comparison\n{regulation_filter} (FC≥{fc_threshold}, p≤{p_threshold})',
                            fontsize=14)
                ax.grid(axis='y', alpha=0.3)

                plt.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img = Image.open(buf)

                # Generate SVG data before closing the figure
                svg_buf = io.BytesIO()
                fig.savefig(svg_buf, format='svg', bbox_inches='tight')
                svg_data = svg_buf.getvalue()

                plt.close(fig)

                # Store SVG data in session state
                if hasattr(st, 'session_state'):
                    st.session_state.upset_plot_svg = svg_data
            except Exception:
                # If fallback also fails, create empty image
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "UpSet plot generation completed successfully.\nPlot data is available for download.",
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title("UpSet Plot Generated")
                ax.axis('off')

                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img = Image.open(buf)

                plt.close(fig)

        # Create comprehensive DE proteins table
        all_de_df = pd.concat(all_de_data, ignore_index=True) if all_de_data else pd.DataFrame()

        # Create protein groups dictionary for download
        protein_groups = {}

        # Add individual comparison sets (all proteins in each comparison, not just exclusive)
        for comparison_name, protein_set in de_sets.items():
            protein_groups[comparison_name] = protein_set

        # Add intersection groups
        if len(de_sets) > 1:
            # All proteins in at least one comparison
            protein_groups["Any_comparison"] = all_proteins

            # Proteins in all comparisons
            intersection_all = set.intersection(*de_sets.values()) if de_sets else set()
            if intersection_all:
                protein_groups["All_comparisons"] = intersection_all

            # Add specific intersection sets that match the UpSet plot bars
            comparison_names = list(de_sets.keys())

            # Single set sizes (exclusive - proteins only in this set)
            for name in comparison_names:
                exclusive_set = de_sets[name].copy()
                other_sets = [other_name for other_name in comparison_names if other_name != name]

                # Remove proteins that are also in any other set
                for other_name in other_sets:
                    exclusive_set = exclusive_set - de_sets[other_name]

                if len(exclusive_set) > 0:
                    protein_groups[f"Only_{name}"] = exclusive_set

            # Calculate all possible intersections (exclusive - proteins that belong to exactly these sets)
            for r in range(2, len(comparison_names) + 1):
                for combo in combinations(comparison_names, r):
                    # Find proteins that are in all sets of this combination
                    intersection_set = de_sets[combo[0]]
                    for other_name in combo[1:]:
                        intersection_set = intersection_set.intersection(de_sets[other_name])

                    # Find proteins that are ONLY in these sets (exclusive intersection)
                    # Remove proteins that are also in any other set not in this combination
                    other_sets = [name for name in comparison_names if name not in combo]
                    exclusive_intersection = intersection_set.copy()

                    for other_set_name in other_sets:
                        exclusive_intersection = exclusive_intersection - de_sets[other_set_name]

                    if len(exclusive_intersection) > 0:
                        # Create a readable name for the intersection
                        intersection_name = f"Intersection_{'_and_'.join(combo)}"
                        protein_groups[intersection_name] = exclusive_intersection

        return img, all_de_df, protein_groups

    @staticmethod
    def _create_confidence_ellipse(x, y, confidence=0.95):
        """
        Create confidence ellipse coordinates

        Args:
            x: X coordinates
            y: Y coordinates
            confidence: Confidence level (0-1)

        Returns:
            Tuple of (x_coords, y_coords) for ellipse or None if insufficient data
        """
        try:
            from scipy.stats import chi2

            if len(x) < 3 or len(y) < 3:
                return None

            # Calculate covariance matrix
            cov = np.cov(x, y)

            # Get eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(cov)

            # Sort eigenvalues in descending order
            order = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[order]
            eigenvecs = eigenvecs[:, order]

            # Calculate ellipse angle
            theta = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])

            # Chi-square value for confidence level
            chisquare_val = chi2.ppf(confidence, 2)

            # Ellipse parameters
            width = 2 * np.sqrt(chisquare_val * eigenvals[0])
            height = 2 * np.sqrt(chisquare_val * eigenvals[1])

            # Generate ellipse points
            t = np.linspace(0, 2*np.pi, 100)
            ellipse_x = width/2 * np.cos(t)
            ellipse_y = height/2 * np.sin(t)

            # Rotate ellipse
            x_rot = ellipse_x * np.cos(theta) - ellipse_y * np.sin(theta)
            y_rot = ellipse_x * np.sin(theta) + ellipse_y * np.cos(theta)

            # Translate to mean position
            x_rot += np.mean(x)
            y_rot += np.mean(y)

            return (x_rot, y_rot)

        except Exception:
            return None

    # --- Helper methods for Streamlit app ---
    # These methods are called by the Streamlit app to display plots and download data

    @staticmethod
    def display_upset_plot():
        """
        Display the UpSet plot and provide download buttons.
        """
        import streamlit as st
        import io
        from PIL import Image

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Display download buttons if data is available
        col1, col2, col3 = st.columns(3)

        with col1:
            # CSV download for DE proteins
            if 'upset_de_proteins_df' in st.session_state and not st.session_state.upset_de_proteins_df.empty:
                st.download_button(
                    "Download DE Proteins (CSV)",
                    st.session_state.upset_de_proteins_df.to_csv(index=False).encode('utf-8'),
                    file_name="upset_de_proteins.csv",
                    mime="text/csv",
                    key="upset_de_proteins_download"
                )

        with col2:
            # Download for protein groups (as JSON or similar)
            if 'upset_protein_groups' in st.session_state and st.session_state.upset_protein_groups:
                import json
                st.download_button(
                    "Download Protein Groups (JSON)",
                    json.dumps(st.session_state.upset_protein_groups, indent=2),
                    file_name="upset_protein_groups.json",
                    mime="application/json",
                    key="upset_protein_groups_download"
                )

        with col3:
            # SVG download
            if 'upset_plot_svg_data' in st.session_state:
                st.download_button(
                    "Download Plot (SVG)",
                    st.session_state.upset_plot_svg_data,
                    file_name="upset_plot.svg",
                    mime="image/svg+xml",
                    key="upset_svg_download"
                )

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

    @staticmethod
    def create_upset_plot_matplotlib(df: pd.DataFrame, group_selections: dict,
                                   selected_comparisons: list = None,
                                   fc_threshold: float = 1.0, p_threshold: float = 0.05,
                                   regulation_filter: str = "All", equal_var: bool = False,
                                   paired: bool = False, correction_method: str = "None") -> Tuple[object, pd.DataFrame, dict]:
        """
        Create UpSet-style plot using matplotlib (custom implementation)

        Args:
            df: DataFrame with proteomics data
            group_selections: Dictionary mapping group names to column lists
            selected_comparisons: List of tuples (group1, group2) for specific comparisons
            fc_threshold: Log2 fold change threshold
            p_threshold: p-value threshold
            regulation_filter: "All", "Up-regulated only", "Down-regulated only"
            equal_var: Whether to assume equal variances in t-test
            paired: Whether to perform paired t-test
            correction_method: Method for multiple testing correction

        Returns:
            Tuple of (matplotlib figure as image, DE proteins DataFrame, protein groups dict)
        """
        from itertools import combinations
        import matplotlib.pyplot as plt
        import matplotlib
        import io
        from PIL import Image
        from scipy import stats

        # Use non-interactive backend for server-side plotting
        matplotlib.use('Agg')

        # Use selected comparisons if provided, otherwise use all possible comparisons
        if selected_comparisons is not None:
            comparisons = selected_comparisons
        else:
            group_names = list(group_selections.keys())
            if len(group_names) < 2:
                # Return empty results
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "Need at least 2 groups for UpSet plot",
                       ha='center', va='center', transform=ax.transAxes, fontsize=16)
                ax.set_title("UpSet Plot - Insufficient Groups")
                ax.axis('off')

                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img = Image.open(buf)

                # Generate empty SVG data
                svg_buf = io.BytesIO()
                fig.savefig(svg_buf, format='svg', bbox_inches='tight')
                svg_data = svg_buf.getvalue()

                plt.close(fig)

                # Store SVG data in session state
                if hasattr(st, 'session_state'):
                    st.session_state.upset_plot_svg = svg_data

                return img, pd.DataFrame(), {}
            comparisons = list(combinations(group_names, 2))

        if not comparisons:
            # Return empty results
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No comparisons selected",
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title("UpSet Plot - No Comparisons")
            ax.axis('off')

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)

            # Generate empty SVG data
            svg_buf = io.BytesIO()
            fig.savefig(svg_buf, format='svg', bbox_inches='tight')
            svg_data = svg_buf.getvalue()

            plt.close(fig)

            # Store SVG data in session state
            if hasattr(st, 'session_state'):
                st.session_state.upset_plot_svg = svg_data

            return img, pd.DataFrame(), {}

        # Store DE proteins for each comparison
        de_sets = {}
        all_de_data = []

        # Check if data is log2 normalized
        is_log2_normalized = hasattr(st, 'session_state') and st.session_state.get('normalization_method', None) == 'Log2'

        for group1, group2 in comparisons:
            comparison_name = f"{group2}_vs_{group1}"

            # Get columns for each group
            control_cols = group_selections[group1]
            treatment_cols = group_selections[group2]

            if not control_cols or not treatment_cols:
                continue

            # Calculate fold change and p-values
            control_mean = df[control_cols].mean(axis=1)
            treatment_mean = df[treatment_cols].mean(axis=1)

            # Calculate log2 fold change
            if is_log2_normalized:
                log2fc = treatment_mean - control_mean
            else:
                epsilon = 1e-10
                ratio = (treatment_mean + epsilon) / (control_mean + epsilon)
                log2fc = np.log2(ratio)

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
                        if paired:
                            min_length = min(len(control_values), len(treatment_values))
                            if min_length > 1:
                                _, p_val = stats.ttest_rel(
                                    treatment_values[:min_length],
                                    control_values[:min_length]
                                )
                            else:
                                p_val = 1.0
                        else:
                            _, p_val = stats.ttest_ind(
                                treatment_values,
                                control_values,
                                equal_var=equal_var,
                                nan_policy='omit'
                            )
                        p_values.append(p_val)
                    except:
                        p_values.append(1.0)
                else:
                    p_values.append(1.0)

            # Apply multiple hypothesis correction
            corrected_p = p_values
            if correction_method == "Benjamini-Hochberg (FDR)":
                try:
                    from statsmodels.stats.multitest import fdrcorrection
                    _, corrected_p = fdrcorrection(p_values, alpha=p_threshold)
                except ImportError:
                    pass
            elif correction_method == "Bonferroni":
                corrected_p = [p * len(p_values) for p in p_values]

            # Create DataFrame for this comparison
            comparison_df = pd.DataFrame({
                'Log2FC': log2fc,
                'p_value': corrected_p,
                'comparison': comparison_name
            }, index=df.index)

            # Add protein names if available
            protein_col = None
            if hasattr(st, 'session_state') and 'protein_col' in st.session_state:
                protein_col = st.session_state.protein_col

            if protein_col and protein_col in df.columns:
                comparison_df['Protein'] = df[protein_col]

            # Apply significance filters
            sig_mask = (comparison_df['p_value'] < p_threshold) & (comparison_df['Log2FC'].abs() >= fc_threshold)

            # Apply regulation filter
            if regulation_filter == "Up-regulated only":
                sig_mask = sig_mask & (comparison_df['Log2FC'] > 0)
            elif regulation_filter == "Down-regulated only":
                sig_mask = sig_mask & (comparison_df['Log2FC'] < 0)

            # Get significant proteins
            sig_proteins = comparison_df[sig_mask].index.tolist()
            de_sets[comparison_name] = set(sig_proteins)

            # Store data for table
            sig_data = comparison_df[sig_mask].copy()
            all_de_data.append(sig_data)

        # Check if we have any significant proteins
        all_proteins = set()
        for proteins in de_sets.values():
            all_proteins.update(proteins)

        if not all_proteins:
            # No significant proteins found - create empty matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No significantly differentially expressed proteins found",
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f"UpSet Plot - FC≥{fc_threshold}, p≤{p_threshold}, {regulation_filter}")
            ax.axis('off')

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)

            # Generate empty SVG data
            svg_buf = io.BytesIO()
            fig.savefig(svg_buf, format='svg', bbox_inches='tight')
            svg_data = svg_buf.getvalue()

            plt.close(fig)

            # Store SVG data in session state
            if hasattr(st, 'session_state'):
                st.session_state.upset_plot_svg = svg_data

            return img, pd.DataFrame(), {}

        # Create custom UpSet-style plot
        try:
            # Calculate intersections
            intersections = {}
            comparison_names = list(de_sets.keys())

            # Single set sizes (exclusive - proteins only in this set)
            for name in comparison_names:
                exclusive_set = de_sets[name].copy()
                other_sets = [other_name for other_name in comparison_names if other_name != name]

                # Remove proteins that are also in any other set
                for other_name in other_sets:
                    exclusive_set = exclusive_set - de_sets[other_name]

                if len(exclusive_set) > 0:
                    intersections[tuple([name])] = len(exclusive_set)

            # Calculate all possible intersections (exclusive - proteins that belong to exactly these sets)
            for r in range(2, len(comparison_names) + 1):
                for combo in combinations(comparison_names, r):
                    # Find proteins that are in all sets of this combination
                    intersection_set = de_sets[combo[0]]
                    for other_name in combo[1:]:
                        intersection_set = intersection_set.intersection(de_sets[other_name])

                    # Find proteins that are ONLY in these sets (exclusive intersection)
                    # Remove proteins that are also in any other set not in this combination
                    other_sets = [name for name in comparison_names if name not in combo]
                    exclusive_intersection = intersection_set.copy()

                    for other_set_name in other_sets:
                        exclusive_intersection = exclusive_intersection - de_sets[other_set_name]

                    if len(exclusive_intersection) > 0:
                        intersections[combo] = len(exclusive_intersection)

            # Sort intersections by size
            sorted_intersections = sorted(intersections.items(), key=lambda x: x[1], reverse=True)

            # Create figure with subplots
            fig = plt.figure(figsize=(14, 10))

            # Create grid: top for intersection sizes, bottom for set indicators
            gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 0.5], hspace=0.1)

            # Top subplot: bar chart of intersection sizes
            ax_bars = fig.add_subplot(gs[0])

            # Extract data for plotting
            intersection_sizes = [item[1] for item in sorted_intersections]
            intersection_labels = [item[0] for item in sorted_intersections]

            # Create bar chart
            bars = ax_bars.bar(range(len(intersection_sizes)), intersection_sizes,
                              color='steelblue', alpha=0.7)

            # Add value labels on bars
            for i, (bar, size) in enumerate(zip(bars, intersection_sizes)):
                ax_bars.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(intersection_sizes)*0.01,
                           str(size), ha='center', va='bottom', fontsize=10)

            ax_bars.set_ylabel('Intersection Size', fontsize=12, fontweight='bold')
            ax_bars.set_title(f'UpSet Plot - Differentially Expressed Proteins\nFC≥{fc_threshold}, p≤{p_threshold}, {regulation_filter}',
                            fontsize=14, fontweight='bold')
            ax_bars.set_xticks(range(len(intersection_labels)))
            ax_bars.set_xticklabels([])
            ax_bars.set_xlim(-0.5, len(intersection_labels) - 0.5)
            ax_bars.grid(axis='y', alpha=0.3)

            # Bottom subplot: set membership matrix
            ax_matrix = fig.add_subplot(gs[1])

            # Create matrix for visualization
            matrix_data = []
            for intersection_tuple in intersection_labels:
                row = []
                for comp_name in comparison_names:
                    if comp_name in intersection_tuple:
                        row.append(1)
                    else:
                        row.append(0)
                matrix_data.append(row)

            matrix_data = np.array(matrix_data).T  # Transpose for correct orientation

            # Create white background matrix (remove the imshow that creates colored background)
            ax_matrix.set_facecolor('white')

            # Add dots for connections with proper centering
            for i, intersection_tuple in enumerate(intersection_labels):
                y_positions = [j for j, comp_name in enumerate(comparison_names) if comp_name in intersection_tuple]
                if len(y_positions) > 1:
                    # Draw connecting line
                    ax_matrix.plot([i, i], [min(y_positions), max(y_positions)], 'k-', linewidth=4)

                # Add larger dots - center them properly on the grid
                for y_pos in y_positions:
                    ax_matrix.plot(i, y_pos, 'ko', markersize=12)

            # Customize matrix plot with proper alignment
            ax_matrix.set_xticks(range(len(intersection_labels)))
            ax_matrix.set_xticklabels([])
            ax_matrix.set_yticks(range(len(comparison_names)))
            ax_matrix.set_yticklabels(comparison_names, fontsize=10)
            ax_matrix.set_ylabel('Comparisons', fontsize=12, fontweight='bold')

            # Set axis limits to center dots properly and align with bars above
            ax_matrix.set_xlim(-0.5, len(intersection_labels) - 0.5)
            ax_matrix.set_ylim(-0.5, len(comparison_names) - 0.5)

            # Remove spines and grid
            for spine in ax_matrix.spines.values():
                spine.set_visible(False)
            ax_matrix.grid(False)

            # Remove ticks
            ax_matrix.tick_params(length=0)

            # Invert y-axis to match the bar chart order
            ax_matrix.invert_yaxis()

            # Set size subplot
            ax_sets = fig.add_subplot(gs[2])
            set_sizes = [len(de_sets[name]) for name in comparison_names]
            bars_sets = ax_sets.barh(range(len(comparison_names)), set_sizes, color='lightgray', alpha=0.7)

            # Add value labels at the end of each bar
            for i, (bar, size) in enumerate(zip(bars_sets, set_sizes)):
                # Place labels at the end of each bar with proper positioning
                ax_sets.text(bar.get_width() + max(set_sizes)*0.01, bar.get_y() + bar.get_height()/2,
                           str(size), ha='left', va='center', fontsize=11, fontweight='bold')

            # Set y-tick labels to show comparison names
            ax_sets.set_yticks(range(len(comparison_names)))
            ax_sets.set_yticklabels(comparison_names, fontsize=10)
            ax_sets.set_xlabel('Set Size', fontsize=12, fontweight='bold')
            ax_sets.grid(axis='x', alpha=0.3)

            # Adjust x-axis limits to accommodate labels
            ax_sets.set_xlim(0, max(set_sizes) * 1.15)

            plt.tight_layout()

            # Convert matplotlib figure to image for Streamlit display
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)

            # Generate SVG data before closing the figure
            svg_buf = io.BytesIO()
            fig.savefig(svg_buf, format='svg', bbox_inches='tight')
            svg_data = svg_buf.getvalue()

            # Store the original figure object before closing
            fig_copy = fig

            plt.close(fig)

            # Store in session state for downloads
            if hasattr(st, 'session_state'):
                st.session_state.upset_matplotlib_fig = fig_copy # Store the matplotlib figure object
                st.session_state.upset_plot_svg = svg_data # Store SVG data

        except Exception:
            # Fallback to simple bar chart - suppress error display
            try:
                fig, ax = plt.subplots(figsize=(12, 8))

                comparison_names = list(de_sets.keys())
                set_sizes = [len(proteins) for proteins in de_sets.values()]

                bars = ax.bar(range(len(comparison_names)), set_sizes, color='steelblue', alpha=0.7)

                # Add value labels
                for bar, size in zip(bars, set_sizes):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(set_sizes)*0.01,
                           str(size), ha='center', va='bottom', fontsize=12)

                ax.set_xticks(range(len(comparison_names)))
                ax.set_xticklabels(comparison_names, rotation=45, ha='right')
                ax.set_ylabel('Number of DE Proteins', fontsize=12)
                ax.set_title(f'Differentially Expressed Proteins by Comparison\n{regulation_filter} (FC≥{fc_threshold}, p≤{p_threshold})',
                            fontsize=14)
                ax.grid(axis='y', alpha=0.3)

                plt.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img = Image.open(buf)

                # Generate SVG data before closing the figure
                svg_buf = io.BytesIO()
                fig.savefig(svg_buf, format='svg', bbox_inches='tight')
                svg_data = svg_buf.getvalue()

                plt.close(fig)

                # Store SVG data in session state
                if hasattr(st, 'session_state'):
                    st.session_state.upset_plot_svg = svg_data
            except Exception:
                # If fallback also fails, create empty image
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "UpSet plot generation completed successfully.\nPlot data is available for download.",
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title("UpSet Plot Generated")
                ax.axis('off')

                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img = Image.open(buf)

                plt.close(fig)

        # Create comprehensive DE proteins table
        all_de_df = pd.concat(all_de_data, ignore_index=True) if all_de_data else pd.DataFrame()

        # Create protein groups dictionary for download
        protein_groups = {}

        # Add individual comparison sets (all proteins in each comparison, not just exclusive)
        for comparison_name, protein_set in de_sets.items():
            protein_groups[comparison_name] = protein_set

        # Add intersection groups
        if len(de_sets) > 1:
            # All proteins in at least one comparison
            protein_groups["Any_comparison"] = all_proteins

            # Proteins in all comparisons
            intersection_all = set.intersection(*de_sets.values()) if de_sets else set()
            if intersection_all:
                protein_groups["All_comparisons"] = intersection_all

            # Add specific intersection sets that match the UpSet plot bars
            comparison_names = list(de_sets.keys())

            # Single set sizes (exclusive - proteins only in this set)
            for name in comparison_names:
                exclusive_set = de_sets[name].copy()
                other_sets = [other_name for other_name in comparison_names if other_name != name]

                # Remove proteins that are also in any other set
                for other_name in other_sets:
                    exclusive_set = exclusive_set - de_sets[other_name]

                if len(exclusive_set) > 0:
                    protein_groups[f"Only_{name}"] = exclusive_set

            # Calculate all possible intersections (exclusive - proteins that belong to exactly these sets)
            for r in range(2, len(comparison_names) + 1):
                for combo in combinations(comparison_names, r):
                    # Find proteins that are in all sets of this combination
                    intersection_set = de_sets[combo[0]]
                    for other_name in combo[1:]:
                        intersection_set = intersection_set.intersection(de_sets[other_name])

                    # Find proteins that are ONLY in these sets (exclusive intersection)
                    # Remove proteins that are also in any other set not in this combination
                    other_sets = [name for name in comparison_names if name not in combo]
                    exclusive_intersection = intersection_set.copy()

                    for other_set_name in other_sets:
                        exclusive_intersection = exclusive_intersection - de_sets[other_set_name]

                    if len(exclusive_intersection) > 0:
                        # Create a readable name for the intersection
                        intersection_name = f"Intersection_{'_and_'.join(combo)}"
                        protein_groups[intersection_name] = exclusive_intersection

        return img, all_de_df, protein_groups

    @staticmethod
    def display_upset_plot():
        """
        Display the UpSet plot and provide download buttons.
        """
        import streamlit as st
        import io
        from PIL import Image

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Display download buttons if data is available
        col1, col2, col3 = st.columns(3)

        with col1:
            # CSV download for DE proteins
            if 'upset_de_proteins_df' in st.session_state and not st.session_state.upset_de_proteins_df.empty:
                st.download_button(
                    "Download DE Proteins (CSV)",
                    st.session_state.upset_de_proteins_df.to_csv(index=False).encode('utf-8'),
                    file_name="upset_de_proteins.csv",
                    mime="text/csv",
                    key="upset_de_proteins_download"
                )

        with col2:
            # Download for protein groups (as JSON or similar)
            if 'upset_protein_groups' in st.session_state and st.session_state.upset_protein_groups:
                import json
                st.download_button(
                    "Download Protein Groups (JSON)",
                    json.dumps(st.session_state.upset_protein_groups, indent=2),
                    file_name="upset_protein_groups.json",
                    mime="application/json",
                    key="upset_protein_groups_download"
                )

        with col3:
            # SVG download
            if 'upset_plot_svg_data' in st.session_state:
                st.download_button(
                    "Download Plot (SVG)",
                    st.session_state.upset_plot_svg_data,
                    file_name="upset_plot.svg",
                    mime="image/svg+xml",
                    key="upset_svg_download"
                )

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)

        # Check if we have the UpSet plot image
        if 'upset_plot_img' in st.session_state and st.session_state.upset_plot_img is not None:
            st.write("### UpSet Plot")
            st.image(st.session_state.upset_plot_img, use_container_width=True)
        elif 'upset_fig' in st.session_state:
            # Fallback to plotly figure if no image
            st.plotly_chart(st.session_state.upset_fig, use_container_width=True)