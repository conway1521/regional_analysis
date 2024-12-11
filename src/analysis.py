import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from factor_analyzer import FactorAnalyzer
from scipy import stats
import networkx as nx
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
import geopandas as gpd
from pysal.lib import weights
from pysal.explore import esda
import plotly.express as px
import plotly.graph_objects as go
import pickle

class RegionalAnalysis:
    def __init__(self, data_path, pillars_dict):
        """
        Initialize analysis with data and pillar structure

        Parameters:
        data_path: Path to CSV with regional data (your eu_nuts2_aligned_data.csv)
        pillars_dict: Dictionary defining pillar-subject-variable structure
        """
        # Load data
        self.df = pd.read_csv(data_path)
        
        # NUTS2 GeoJSON URL - stored as class attribute
        self.GEOJSON_URL = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_20M_2021_3035_LEVL_2.geojson"
        
        self.pillars = pillars_dict
        self.scaler = StandardScaler()

    def preprocess_data(self):
        """
        Standardize variables and handle missing values
        Returns: Preprocessed DataFrame
        """
        # Get all variables from pillars dictionary
        variables = []
        for pillar, subjects in self.pillars.items():
            for subject, vars_list in subjects.items():
                variables.extend(vars_list)

        # Standardize variables
        X = self.df[variables]
        X_scaled = self.scaler.fit_transform(X)
        self.df_scaled = pd.DataFrame(X_scaled, columns=variables)

        return self.df_scaled

    def dimension_reduction(self):
        """
        Perform PCA and Varimax rotation
        Returns: DataFrame with principal components
        """
        # Perform PCA first to determine number of factors
        pca = PCA(n_components=0.95)  # Keep 95% of variance
        pca_result = pca.fit_transform(self.df_scaled)
        n_components = pca_result.shape[1]

        # Perform factor analysis with determined number of components
        fa = FactorAnalyzer(rotation='varimax', n_factors=n_components)
        fa.fit(self.df_scaled)

        # Get loadings
        loadings = pd.DataFrame(
            fa.loadings_,
            columns=[f'Factor{i + 1}' for i in range(n_components)],
            index=self.df_scaled.columns
        )

        self.loadings = loadings
        return loadings

    def feature_selection(self):
        """
        Use Random Forests to identify important variables
        Returns: DataFrame with variable importance scores
        """
        importance_scores = {}

        # For each pillar
        for pillar, subjects in self.pillars.items():
            pillar_vars = []
            for vars_list in subjects.values():
                pillar_vars.extend(vars_list)

            # Use mean of other variables as target
            X = self.df_scaled[pillar_vars]
            y = X.mean(axis=1)

            # Fit Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)

            # Store importance scores
            for var, imp in zip(pillar_vars, rf.feature_importances_):
                importance_scores[var] = imp

        self.importance = pd.DataFrame.from_dict(
            importance_scores,
            orient='index',
            columns=['Importance']
        ).sort_values('Importance', ascending=False)

        return self.importance

    def create_composite_indices(self):
        """
        Create weighted composite indices for each subject
        Returns: DataFrame with composite indices
        """
        print("\nStarting Composite Indices Creation:")
        print("Original DataFrame head:")
        print(self.df.head())

        indices = {}

        # For each pillar and subject
        for pillar, subjects in self.pillars.items():
            for subject, variables in subjects.items():
                # Get importance weights
                weights = self.importance.loc[variables, 'Importance']
                weights = weights / weights.sum()  # Normalize

                # Calculate weighted sum
                indices[subject] = (
                        self.df_scaled[variables] * weights
                ).sum(axis=1)

        # Create DataFrame with indices
        self.composite_indices = pd.DataFrame(indices)

        # Set NUTS2 code as index
        self.composite_indices.index = self.df['nuts2_code']

        print("\nComposite Indices Created:")
        print(f"Shape: {self.composite_indices.shape}")
        print("\nIndex Preview:")
        print(self.composite_indices.index[:5])

        return self.composite_indices

    def network_analysis(self):
        """
        Analyze relationships between indices using network analysis with enhanced diagnostics
        Returns: NetworkX graph, key metrics, and diagnostic information
        """
        # Create correlation matrix
        corr_matrix = self.composite_indices.corr()

        # Add diagnostic print for correlation matrix
        print("\nCorrelation Matrix Summary:")
        print(f"Shape: {corr_matrix.shape}")
        print("\nCorrelation Range:")
        print(f"Min correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].min():.3f}")
        print(f"Max correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].max():.3f}")

        # Create network
        G = nx.Graph()

        # Add all nodes first (this ensures isolated nodes are included)
        G.add_nodes_from(corr_matrix.columns)

        # Dictionary to store correlation distribution
        corr_distribution = {
            '0.0-0.1': 0,
            '0.1-0.2': 0,
            '0.2-0.3': 0,
            '0.3-0.4': 0,
            '0.4-0.5': 0,
            '0.5+': 0
        }

        # Adjustable correlation threshold - can be modified based on data characteristics
        CORRELATION_THRESHOLD = 0.1  # Lowered from 0.3 to capture more relationships

        # Add edges for significant correlations with enhanced tracking
        edge_count = 0
        significant_correlations = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                abs_corr = abs(corr)

                # Track correlation distribution
                if abs_corr < 0.1:
                    corr_distribution['0.0-0.1'] += 1
                elif abs_corr < 0.2:
                    corr_distribution['0.1-0.2'] += 1
                elif abs_corr < 0.3:
                    corr_distribution['0.2-0.3'] += 1
                elif abs_corr < 0.4:
                    corr_distribution['0.3-0.4'] += 1
                elif abs_corr < 0.5:
                    corr_distribution['0.4-0.5'] += 1
                else:
                    corr_distribution['0.5+'] += 1

                if abs_corr > CORRELATION_THRESHOLD:
                    G.add_edge(
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        weight=abs_corr
                    )
                    edge_count += 1
                    significant_correlations.append({
                        'node1': corr_matrix.columns[i],
                        'node2': corr_matrix.columns[j],
                        'correlation': corr
                    })

        # Print diagnostic information
        print("\nNetwork Construction Summary:")
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {edge_count}")
        print("\nCorrelation Distribution:")
        for range_label, count in corr_distribution.items():
            print(f"{range_label}: {count}")

        # Calculate basic network metrics with safety checks
        metrics = {
            'density': nx.density(G),
            'edge_count': edge_count,
            'average_degree': 2 * edge_count / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
            'isolated_nodes': list(nx.isolates(G))
        }

        # Calculate advanced metrics only if there are sufficient edges
        if edge_count > 0:
            try:
                metrics['clustering'] = nx.average_clustering(G)
                metrics['transitivity'] = nx.transitivity(G)

                # Calculate node-level metrics
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)

                metrics['node_metrics'] = {
                    node: {
                        'degree_centrality': degree_centrality[node],
                        'betweenness_centrality': betweenness_centrality[node]
                    } for node in G.nodes()
                }

                # Try to calculate eigenvector centrality (might fail for disconnected graphs)
                try:
                    metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G)
                except:
                    print("Warning: Could not calculate eigenvector centrality")
                    metrics['eigenvector_centrality'] = {node: 0 for node in G.nodes()}

            except Exception as e:
                print(f"Warning: Error calculating some network metrics: {str(e)}")
                metrics.update({
                    'clustering': 0,
                    'transitivity': 0,
                    'node_metrics': {node: {'degree_centrality': 0, 'betweenness_centrality': 0}
                                     for node in G.nodes()},
                    'eigenvector_centrality': {node: 0 for node in G.nodes()}
                })

        # Store strongest correlations for analysis
        metrics['significant_correlations'] = sorted(
            significant_correlations,
            key=lambda x: abs(x['correlation']),
            reverse=True
        )[:10]  # Store top 10 strongest correlations

        # Add diagnostic information to metrics
        metrics['diagnostics'] = {
            'correlation_distribution': corr_distribution,
            'correlation_threshold_used': CORRELATION_THRESHOLD,
            'isolated_nodes_count': len(metrics['isolated_nodes'])
        }

        self.network = G
        self.network_metrics = metrics

        # Print summary of strongest correlations
        print("\nTop 5 Strongest Correlations:")
        for i, corr in enumerate(metrics['significant_correlations'][:5], 1):
            print(f"{i}. {corr['node1']} -- {corr['node2']}: {corr['correlation']:.3f}")

        return G, metrics

    def spatial_analysis(self):
        """
        Perform spatial econometric analysis with KNN weights
        """
        # Load GeoJSON using class attribute
        gdf = gpd.read_file(self.GEOJSON_URL)

        print("\nSpatial Analysis Dataset Info:")
        print(f"Number of regions in GeoJSON: {len(gdf)}")
        print(f"Number of regions in composite indices: {len(self.composite_indices)}")

        # Merge with data
        gdf = gdf.merge(
            self.composite_indices,
            left_on='NUTS_ID',
            right_index=True,
            how='inner',
            validate='1:1'
        )

        print(f"\nFinal merged dataset has {len(gdf)} regions")

        # Create KNN weights matrix
        try:
            # Use k=4 to ensure each region has 4 neighbors
            w = weights.KNN.from_dataframe(
                gdf,
                k=4,
                use_index=True,
                geom_col='geometry'
            )
            w.transform = 'r'  # Row-standardize weights

            # Print connectivity information
            print("\nSpatial Weights Information:")
            print(f"Number of regions: {len(w.neighbors)}")
            print(f"Number of neighbors per region: {w.cardinalities}")

            # Add basic connectivity check
            print("\nConnectivity Check:")
            print(f"All regions have exactly 4 neighbors: {all(n == 4 for n in w.cardinalities.values())}")
            print(f"Total number of connections: {sum(w.cardinalities.values())}")

        except Exception as e:
            print(f"\nError creating weights matrix: {str(e)}")
            return None

        # Calculate spatial statistics
        spatial_stats = {}

        print("\nCalculating Spatial Statistics:")
        for col in self.composite_indices.columns:
            try:
                # Global Moran's I
                moran = esda.Moran(gdf[col], w)

                # Local Moran's I
                local_moran = esda.Moran_Local(gdf[col], w)

                # Store results
                spatial_stats[col] = {
                    'moran_i': moran.I,
                    'moran_p': moran.p_sim,
                    'local_moran': local_moran.Is,
                    'local_moran_p': local_moran.p_sim,
                    'moran_z_score': moran.z_sim
                }

                # Print summary statistics
                print(f"\n{col}:")
                print(f"Global Moran's I: {moran.I:.3f}")
                print(f"P-value: {moran.p_sim:.3f}")
                print(f"Z-score: {moran.z_sim:.3f}")

            except Exception as e:
                print(f"\nError calculating spatial statistics for {col}: {str(e)}")
                spatial_stats[col] = {
                    'moran_i': None,
                    'moran_p': None,
                    'local_moran': None,
                    'local_moran_p': None,
                    'moran_z_score': None
                }

        self.spatial_stats = spatial_stats
        self.spatial_gdf = gdf

        return spatial_stats

    def create_visualizations(self):
        """
        Create comprehensive visualizations of results
        Returns: Dictionary of plotly figures
        """
        figs = {}

        # 1. Network Graph
        pos = nx.spring_layout(self.network)
        edge_trace = go.Scatter(
            x=[], y=[], line=dict(width=0.5, color='#888'),
            hoverinfo='none', mode='lines'
        )

        for edge in self.network.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        node_trace = go.Scatter(
            x=[], y=[], text=[], mode='markers+text',
            hoverinfo='text', marker=dict(size=20)
        )

        for node in self.network.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([node])

        fig_network = go.Figure(data=[edge_trace, node_trace],
                                layout=go.Layout(
                                    showlegend=False,
                                    hovermode='closest',
                                    margin=dict(b=0, l=0, r=0, t=0)
                                ))

        figs['network'] = fig_network

        # 2. Correlation Heatmap
        fig_corr = px.imshow(
            self.composite_indices.corr(),
            title='Correlation Between Composite Indices'
        )

        figs['correlation'] = fig_corr

        # 3. Variable Importance
        fig_importance = px.bar(
            self.importance,
            title='Variable Importance Scores'
        )

        figs['importance'] = fig_importance

        return figs

    def run_analysis(self):
        """
        Run complete analysis pipeline
        Returns: Dictionary with all results
        """
        results = {
            'preprocessed_data': self.preprocess_data(),
            'dimension_reduction': self.dimension_reduction(),
            'feature_importance': self.feature_selection(),
            'composite_indices': self.create_composite_indices(),
            'network_analysis': self.network_analysis(),
            'spatial_analysis': self.spatial_analysis(),
            'visualizations': self.create_visualizations()
        }
        return results

    def save_results(self, output_path):
        """Save analysis results to pickle file for Jupyter exploration"""
        import pickle  # Add import at top of file if not already there

        # Create results dict with safety checks
        results = {
            'composite_indices': getattr(self, 'composite_indices', None),
            'network_metrics': getattr(self, 'network_metrics', None),
            'network': getattr(self, 'network', None),
            'loadings': getattr(self, 'loadings', None),
            'importance': getattr(self, 'importance', None)
        }

        # Only add spatial results if they exist
        if hasattr(self, 'spatial_stats'):
            results['spatial_stats'] = self.spatial_stats

        if hasattr(self, 'spatial_gdf'):
            results['spatial_gdf'] = self.spatial_gdf

        print("\nSaving results:")
        print("Available components:", list(results.keys()))

        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
