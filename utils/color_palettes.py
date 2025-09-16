"""
Color palettes for the application.

This module contains color-blind friendly palettes for various visualization types.
"""

import plotly.express as px
import plotly.graph_objects as go

class ColorPalettes:
    """
    Provides color-blind friendly palettes for various visualization types.
    """
    
    # Color-blind friendly palette based on ColorBrewer and Okabe-Ito schemes
    # These are suitable for most types of color vision deficiency
    COLORBLIND_FRIENDLY = [
        "#0072B2",  # Blue
        "#D55E00",  # Vermillion
        "#009E73",  # Green
        "#CC79A7",  # Pink
        "#56B4E9",  # Light blue
        "#E69F00",  # Orange
        "#F0E442",  # Yellow
        "#999999",  # Gray
    ]
    
    # Color-blind friendly diverging palette for heatmaps (blue to white to red)
    COLORBLIND_DIVERGING = [
        "#1965B0",  # Dark blue
        "#5289C7",  # Medium blue
        "#7BAFDE",  # Light blue
        "#FFFFFF",  # White
        "#EE8866",  # Light red/orange
        "#DC5A5A",  # Medium red
        "#B8181B",  # Dark red
    ]
    
    # Sequential palette (light to dark)
    COLORBLIND_SEQUENTIAL = [
        "#edf8fb",
        "#b3cde3",
        "#8c96c6",
        "#8856a7",
        "#810f7c"
    ]
    
    # Categorical palette specifically designed for protanopia/deuteranopia (red-green color blindness)
    PROTANOPIA_DEUTERANOPIA = [
        "#0072B2",  # Blue
        "#E69F00",  # Orange
        "#56B4E9",  # Light blue
        "#D55E00",  # Vermillion
        "#8C6BB1",  # Purple
        "#009E73",  # Green
        "#999999",  # Gray
        "#CC79A7",  # Pink
    ]
    
    # Palette for tritanopia (blue-yellow color blindness)
    TRITANOPIA = [
        "#DC267F",  # Magenta
        "#648FFF",  # Blue
        "#FE6100",  # Orange
        "#785EF0",  # Purple
        "#FFB000",  # Gold
        "#018571",  # Teal
        "#999999",  # Gray
    ]
    
    @staticmethod
    def set_default_plotly_colors():
        """Set the default plotly colors to colorblind-friendly palette"""
        px.defaults.color_continuous_scale = px.colors.sequential.Viridis
        px.defaults.color_discrete_sequence = ColorPalettes.COLORBLIND_FRIENDLY
        
    @staticmethod
    def get_colorscale_for_heatmap():
        """Get a colorblind-friendly colorscale for heatmaps"""
        return [
            [0.0, "#1965B0"],  # Dark blue for low values
            [0.25, "#5289C7"],  # Medium blue
            [0.5, "#FFFFFF"],  # White for middle values
            [0.75, "#EE8866"],  # Light red/orange
            [1.0, "#B8181B"],  # Dark red for high values
        ]

    @staticmethod
    def get_colorscale_for_correlation():
        """Get a colorblind-friendly colorscale for correlation heatmaps"""
        return [
            [0.0, "#1965B0"],  # Dark blue for negative correlation
            [0.5, "#FFFFFF"],  # White for no correlation
            [1.0, "#B8181B"],  # Dark red for positive correlation
        ]
        
    @staticmethod 
    def create_highlighted_palette(base_color="#0072B2", highlight_color="#D55E00", removed_color="#999999"):
        """Create a palette for plots that highlight proteins (kept, removed, highlighted)"""
        return {
            "base": base_color,  # Standard items
            "highlight": highlight_color,  # Highlighted items
            "removed": removed_color,  # Removed/filtered items
        }