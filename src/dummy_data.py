import pandas as pd
import numpy as np
import geopandas as gpd

# EU-27 country codes
eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GR',
                'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO',
                'SE', 'SI', 'SK']

# Create NUTS2 codes (simplified version - 242 regions)
nuts2_codes = []
for country in eu_countries:
    num_regions = np.random.randint(5, 15)  # Random number of NUTS2 regions per country
    for i in range(num_regions):
        nuts2_codes.append(f"{country}{str(i + 1).zfill(2)}")

# Create pillar structure
pillars = {
    'Place_based_Conditions': {
        'Basic_Needs': ['Environmental_Health', 'Food_Physical_Health', 'Housing',
                        'Transportation'],
        'Access': ['Financial_Institution_Access', 'Insurance_Access', 'Broadband_Access',
                   'Research_University_Density', 'RD_Facility_Density', 'Community_College_Density']
    },
    'Human_Social_Capital': {
        'Education_Talent': ['No_Highschool_Completion', 'Mean_Years_Schooling',
                             'Preschool_Enrollment', 'Juvenile_Felony_Rates',
                             'Higher_Ed_Enrollment', 'Disconnected_Youth'],
        'Social_Capital': ['Economic_Connectedness', 'Social_Clustering',
                           'Support_Network_Ratios', 'Civic_Organization_Density']
    },
    'Economic_Activity': {
        'Growth_Prosperity': ['Real_GDP', 'GDP_Growth', 'GDP_per_capita', 'Jobs_Productivity',
                              'Worker_Productivity'],
        'Labor_Market': ['Total_Jobs', 'Youth_Employment', 'Employment_Population_Ratio',
                         'High_Skill_Jobs', 'Overall_Unemployment', 'Youth_Unemployment_Gap',
                         'Labor_Force_Participation', 'Youth_Participation_Gap',
                         'Median_Earnings', 'Gender_Earnings_Gap']
    }
}

# Create dataframe
data = []
for nuts2 in nuts2_codes:
    row = {'nuts2_code': nuts2, 'country': nuts2[:2]}

    # Generate random values for each variable
    for pillar, subjects in pillars.items():
        for subject, variables in subjects.items():
            for var in variables:
                row[var] = np.random.uniform(0, 100)

    data.append(row)

df = pd.DataFrame(data)

# Save to CSV
# df.to_csv('/Users/aconway/Dropbox/PhD/Research/Paper 1/Analysis/code/app/data/eu_nuts2_dummy_data.csv', index=False)

# Print first few rows and structure
print("\nDataframe Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nPillar structure saved in 'pillars' dictionary")


def align_nuts2_data():
    """
    Align dummy data with actual NUTS2 regions
    """
    # Read GeoJSON
    GEOJSON_URL = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_20M_2021_3035_LEVL_2.geojson"
    gdf = gpd.read_file(GEOJSON_URL)
    #gdf = gpd.read_file(geojson_path)

    # Filter for NUTS2 level and EU countries only
    nuts2_gdf = gdf[gdf['LEVL_CODE'] == 2].copy()
    eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI',
                   'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT',
                   'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
    nuts2_gdf = nuts2_gdf[nuts2_gdf['CNTR_CODE'].isin(eu_countries)]

    # Read dummy data
    dummy_df = df

    # Get actual NUTS2 codes
    real_nuts2_codes = nuts2_gdf['NUTS_ID'].tolist()

    # Create new dataframe with real NUTS2 codes
    columns = dummy_df.columns.tolist()
    columns.remove('nuts2_code')  # Remove old NUTS2 code column

    # Generate random data for actual number of regions
    new_data = []
    for nuts2 in real_nuts2_codes:
        row = {'nuts2_code': nuts2, 'country': nuts2[:2]}
        for col in columns:
            if col != 'country':
                row[col] = np.random.uniform(0, 100)
        new_data.append(row)

    aligned_df = pd.DataFrame(new_data)
    return aligned_df

# Generate new aligned dataset
aligned_df = align_nuts2_data()
aligned_df.to_csv('/Users/ali/Dropbox/PhD/Research/Paper_1/Analysis/code/app/data/eu_nuts2_aligned_data.csv', index=False)

print(f"Created dataset with {len(aligned_df)} NUTS2 regions")
print("\nFirst few rows:")
print(aligned_df.head())