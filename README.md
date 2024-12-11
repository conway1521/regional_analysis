# EU Regional Development Analysis Framework

## Overview
This repository contains a comprehensive analytical framework for studying regional development across EU NUTS2 regions. The framework integrates three key pillars:
- Place-based Conditions
- Human and Social Capital
- Economic Activity

The analysis combines traditional economic metrics with social and institutional factors to provide a more nuanced understanding of regional development patterns.

## Project Structure
```
eu-regional-development/
├── data/               # Regional datasets and GeoJSON files
├── notebooks/         # Jupyter notebooks for exploration
├── output/         # Jupyter notebooks for exploration
├── src/              # Core analysis modules
│   ├── analysis.py   # Main analysis pipeline
│   ├── pillars_structure.py # Framework structure
│   └── visualization.py # Visualization tools
├── scripts/          # Executable scripts
│   └── run_analysis.py
└── requirements.txt  # Project dependencies
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/[your-username]/eu-regional-development.git
cd eu-regional-development
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
To run the complete analysis pipeline:
```bash
python scripts/run_analysis.py
```

The analysis will:
1. Process regional data across all three pillars
2. Generate composite indices
3. Perform network and spatial analysis
4. Create visualizations
5. Save results for further exploration

Results can be explored using the Jupyter notebooks in the `notebooks/` directory.

## Data Requirements
The analysis requires:
- NUTS2 level socioeconomic data
- NUTS2 GeoJSON files for spatial analysis
- Data should be stored in the `data/` directory

## Output
The analysis generates:
- Composite indices for each pillar
- Network analysis of inter-relationships
- Spatial analysis results
- Visualization of regional patterns

Results are saved as pickle files for further analysis in Jupyter notebooks.


The approach aims to provide a more comprehensive understanding of regional development dynamics by considering both traditional economic metrics and broader social/institutional factors.
