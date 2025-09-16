
# Proteomics Data Analysis Platform

A comprehensive web application for advanced proteomics data analysis, built with Streamlit. This platform provides an end-to-end workflow for processing, analyzing, and visualizing proteomics datasets with interactive statistical tools and publication-quality visualizations.

## Features

### Data Processing & Analysis
- **Automatic Data Loading**: Loads comprehensive example proteomics data automatically for immediate analysis
- **Advanced Data Processing**: Normalization (Log2), filtering, imputation methods, and quality control tools
- **Statistical Analysis**: T-tests (paired/unpaired), ANOVA, correlation analysis, PLS-DA, and multiple comparison corrections
- **Power Analysis**: Statistical power calculations for study design validation

### Interactive Visualizations
- **Volcano Plots**: With customizable thresholds and statistical annotations
- **PCA Analysis**: Principal component analysis with confidence ellipses and clustering
- **PLS-DA Analysis**: Supervised analysis with VIP (Variable Importance in Projection) scoring
- **Heatmaps**: Both detailed sample-level and group average visualizations
- **Custom Protein Heatmaps**: User-specified protein subset analysis
- **UpSet Plots**: Visualize overlapping differentially expressed proteins across comparisons
- **Correlation Plots**: Within-group and between-group correlation analysis
- **Protein Expression Bar Plots**: Individual protein expression with statistical comparisons
- **Protein Rank Plots**: Dynamic range visualization of the proteome
- **Intensity Histograms**: Distribution analysis by sample groups

### Advanced Features
- **Multiple Testing Corrections**: Benjamini-Hochberg (FDR), Bonferroni, and permutation tests
- **Filtering Options**: By peptide count, coefficient of variation, and valid values percentage
- **Imputation Methods**: Mean, median, minimum, KNN, group-wise mean, and truncated normal
- **Export Capabilities**: Download visualizations as interactive HTML, publication-quality SVG, or CSV data files
- **Color-blind Friendly**: Accessible color palettes throughout the application
- **Comprehensive Download Options**: Protein sets, statistical results, and visualization files

## Quick Start

### Online Demo
Access the live application on Replit: [Your Replit URL]

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
streamlit run main.py --server.address 0.0.0.0 --server.port 8501
```

The app will be available at `http://0.0.0.0:8501`

## Usage Workflow

1. **Data Configuration** (Page 1): The app automatically loads example proteomics data with 500 proteins and multiple sample groups
2. **Data Processing** (Page 2): Apply normalization, filtering, and quality control steps with real-time previews
3. **Visualization** (Page 3): Generate interactive plots, perform statistical analyses, and download results

### Sample Groups
The example dataset includes multiple cell line groups perfect for demonstrating comparative proteomics workflows:
- Control samples
- Treatment conditions
- Time-course data
- Biological replicates

## Application Structure

### Main Application
- `main.py` - Main application entry point and welcome page
- `launcher.py` - Standalone launcher for desktop deployment

### Multi-page Interface
- `pages/1_📥_Data_Upload.py` - Data configuration and sample group setup
- `pages/2_🧪_Data_Processing.py` - Data preprocessing, filtering, and normalization
- `pages/3_📈_Visualization.py` - Interactive visualizations and statistical analysis

### Core Modules
- `utils/data_processor.py` - Data preprocessing and filtering functions
- `utils/visualizations.py` - Comprehensive visualization generation with matplotlib and plotly
- `utils/statistics.py` - Statistical analysis functions including power analysis and PLS-DA
- `utils/color_palettes.py` - Color-blind friendly palettes for accessibility

### Data
- `data/ExampleData.xlsx` - Comprehensive example proteomics dataset with multiple sheets

## Technical Features

### Statistical Methods
- Independent and paired t-tests with equal/unequal variance assumptions
- ANOVA with post-hoc comparisons
- Pearson correlation analysis
- PLS-DA with cross-validation and VIP scoring
- Power analysis with effect size calculations
- Multiple testing corrections (FDR, Bonferroni, permutation)

### Visualization Capabilities
- Interactive plotly-based plots with hover information
- Publication-ready matplotlib figures
- SVG export for vector graphics
- Customizable significance thresholds
- Protein highlighting and annotation features
- Real-time parameter adjustment

### Data Processing Pipeline
- Log2 normalization with missing value handling
- Multiple imputation strategies
- Peptide count filtering with flexible criteria
- Coefficient of variation filtering
- Valid values percentage filtering
- Data quality assessment tools

## Dependencies

- Python 3.11+
- Streamlit ≥1.43.1
- Pandas ≥2.2.3
- NumPy ≥1.24.3
- Plotly ≥6.0.0
- SciPy ≥1.15.2
- Matplotlib ≥3.10.1
- Seaborn ≥0.13.2
- Scikit-learn ≥1.4.0
- OpenPyXL ≥3.1.5
- Statsmodels ≥0.14.0
- Kaleido (for image export)

## Deployment

### Replit Deployment
The application is optimized for deployment on Replit with automatic dependency management and port configuration.

### Local Deployment
For local deployment, ensure all dependencies are installed and run with the provided command including proper server address binding.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact: Dr. Nicholas Woods (nicholas.woods@unmc.edu)

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the interactive web interface
- Uses color-blind friendly palettes based on ColorBrewer and Okabe-Ito schemes
- Designed specifically for proteomics research workflows
- Implements best practices for statistical analysis in quantitative proteomics
- Optimized for both exploratory data analysis and publication-ready results
