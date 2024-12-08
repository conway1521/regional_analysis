import sys
from pathlib import Path

# Add the app directory to Python path
APP_DIR = Path(__file__).parent / "app"
sys.path.append(str(APP_DIR))

from src.analysis import RegionalAnalysis, GEOJSON_URL
from src.pillars_structure import PILLARS

# Set up base directory (this should point to app/)
BASE_DIR = Path(__file__).parent / "app"
DATA_DIR = BASE_DIR / "data"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"  # Add this line
DATA_PATH = DATA_DIR / "eu_nuts2_aligned_data.csv"

def main():
    # Initialize analysis
    analysis = RegionalAnalysis(str(DATA_PATH), PILLARS)

    # Run complete analysis pipeline using the URL instead of local file
    results = analysis.run_analysis(GEOJSON_URL)

    # Save results for Jupyter exploration - now in notebooks directory
    results_path = NOTEBOOKS_DIR / 'analysis_results.pkl'
    analysis.save_results(str(results_path))

    # Access different components
    print("Analysis components:")
    for key in results.keys():
        print(f"- {key}")

    # Example: Print first few composite indices
    print("\nComposite Indices preview:")
    print(results['composite_indices'].head())

if __name__ == "__main__":
    main()