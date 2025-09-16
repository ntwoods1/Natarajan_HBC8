# Proteomics Data Analysis Platform

A comprehensive web application for advanced proteomics data analysis, built with Streamlit. This platform provides an end-to-end workflow for processing, analyzing, and visualizing proteomics datasets with interactive statistical tools and publication-quality visualizations.

## Features

- **Automatic Data Loading**: Loads example proteomics data automatically for immediate analysis
- **Interactive Visualizations**: Volcano plots, heatmaps, PCA analysis, correlation matrices, and more
- **Statistical Analysis**: T-tests, ANOVA, correlation analysis with multiple comparison corrections
- **Data Processing**: Normalization, filtering, imputation, and quality control tools
- **Export Capabilities**: Download visualizations as HTML or SVG files
- **Color-blind Friendly**: Accessible color palettes throughout the application

## Quick Start

### Online Demo
Try the live demo at: [Your Replit URL]

### Local Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/proteomics-analysis-platform.git
cd proteomics-analysis-platform
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main.py
```

The app will be available at `http://localhost:8501`

## Usage

1. **Data Upload**: The app automatically loads example proteomics data
2. **Data Processing**: Apply normalization, filtering, and quality control steps
3. **Visualization**: Generate interactive plots and statistical analyses
4. **Export**: Download results for publication or further analysis

## Application Structure

- `main.py` - Main application entry point
- `pages/` - Multi-page Streamlit application
  - `1_📥_Data_Upload.py` - Data upload and validation
  - `2_🧪_Data_Processing.py` - Data preprocessing and filtering
  - `3_📈_Visualization.py` - Interactive visualizations and analysis
- `utils/` - Core analysis modules
  - `data_processor.py` - Data preprocessing functions
  - `visualizations.py` - Visualization generation
  - `statistics.py` - Statistical analysis functions
  - `color_palettes.py` - Color-blind friendly palettes
- `data/` - Example datasets

## Dependencies

- Python 3.11+
- Streamlit
- Pandas
- NumPy
- Plotly
- SciPy
- Matplotlib
- Seaborn
- Scikit-learn
- OpenPyXL

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Uses color-blind friendly palettes based on ColorBrewer and Okabe-Ito schemes
- Designed for proteomics research workflows