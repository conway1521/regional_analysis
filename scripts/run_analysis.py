import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Create output directory if it doesn't exist
output_dir = project_root / "outputs"
output_dir.mkdir(exist_ok=True)

from src.analysis import RegionalAnalysis
from src.pillars_structure import PILLARS

def main():
    # File path to your data
    DATA_PATH = project_root / "data" / "eu_nuts2_aligned_data.csv"
    OUTPUT_PATH = output_dir / "analysis_results.pkl"

    # Initialize analysis
    analysis = RegionalAnalysis(DATA_PATH, PILLARS)

    # Run complete analysis pipeline
    results = analysis.run_analysis()

    # Save results to outputs directory
    analysis.save_results(OUTPUT_PATH)

if __name__ == "__main__":
    main()