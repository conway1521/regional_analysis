from src.analysis import RegionalAnalysis
from src.pillars_structure import PILLARS
from pathlib import Path

# Set up base directory
BASE_DIR = Path(__file__).parent

# Define paths relative to base directory
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "eu_nuts2_aligned_data.csv"
GEOJSON_PATH = DATA_DIR / "NUTS_RG_01M_2024_3035.geojson"


# File paths
# DATA_PATH = "/Users/ali/Dropbox/PhD/Research/Paper_1/Analysis/code/regional_analysis/app/data/eu_nuts2_aligned_data.csv"
# GEOJSON_PATH = "/Users/aconway/Dropbox/PhD/Research/Paper_1/Analysis/code/regional_analysis/app/data/NUTS_RG_01M_2024_3035.geojson"


def main():
    # Initialize analysis
    analysis = RegionalAnalysis(DATA_PATH, PILLARS)

    # Run complete analysis pipeline
    results = analysis.run_analysis(GEOJSON_PATH)

    # Save results for Jupyter exploration
    analysis.save_results('analysis_results.pkl')

    # Access different components
    print("Analysis components:")
    for key in results.keys():
        print(f"- {key}")

    # Example: Print first few composite indices
    print("\nComposite Indices preview:")
    print(results['composite_indices'].head())


if __name__ == "__main__":
    main()