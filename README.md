# EU Regional Development Analysis

Analysis of inclusive economic development across EU NUTS2 regions, applying the same three-pillar framework (Place-based Conditions, Human and Social Capital, Economic Activity) developed for the NED Dashboard to the European regional context.

The goal is to go beyond standard convergence metrics and capture the structural conditions that differentiate durable regional growth from extractive or fragile growth. Output includes composite pillar indices, inter-pillar network analysis, and choropleth maps at the NUTS2 level.

## Methods

The pipeline ingests NUTS2 socioeconomic data from Eurostat, merges it with NUTS2 GeoJSON boundaries, constructs composite indices for each pillar using weighted aggregation, and runs network analysis to measure inter-pillar interdependencies across regions.

## Structure

```
regional_analysis/
├── data/                  # NUTS2 datasets and GeoJSON (not committed)
├── notebooks/             # Exploratory analysis
├── output/                # Generated indices and figures
└── src/
    ├── analysis.py        # Main pipeline
    ├── pillars_structure.py
    └── visualization.py
```

## Setup

```bash
git clone https://github.com/conway1521/regional_analysis.git
cd regional_analysis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_analysis.py
```

This generates composite indices for each pillar, network analysis of inter-relationships, and spatial visualizations saved to `output/`. Results are also saved as pickle files for further exploration in the notebooks.

## Data Requirements

- NUTS2 socioeconomic indicators from Eurostat (download separately)
- NUTS2 boundary GeoJSON from the [Eurostat GISCO service](https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts)
- Place both in `data/`
