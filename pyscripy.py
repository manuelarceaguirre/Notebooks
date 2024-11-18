# # PowerShell Script to Install Required Python Libraries with pip

# # Ensure pip is updated
# Write-Host "Updating pip to the latest version..."
# python -m pip install --upgrade pip

# # Install each required library
# Write-Host "Installing libraries..."
# python -m pip install evidently
# python -m pip install pandas
# python -m pip install numpy
# python -m pip install scipy
# python -m pip install scikit-learn
# python -m pip install shap
# python -m pip install tqdm
# python -m pip install joblib

# Write-Host "All libraries have been installed successfully!"


from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.metrics import *
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfRows, TestColumnsType, TestShareOfMissingValues
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import shap
from sklearn.inspection import permutation_importance
from tqdm import tqdm
import time
from joblib import parallel_backend
import warnings
from functools import partial
from concurrent.futures import TimeoutError
import signal
from datetime import datetime
import os
from evidently.pipeline.column_mapping import ColumnMapping
from scipy import stats

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)

class EvidentlyAssistant:
    def __init__(self, data: pd.DataFrame, target: str, reference_size: float = 0.7):
        """Initialize the assistant with data and configuration"""
        self.data = data
        self.target = target
        self.reference_size = reference_size
        
        # Determine target type
        self.target_type = self._determine_target_type()
        
        # Split numerical and categorical columns
        self.numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Split reference and current datasets
        self._split_data()
        
    def _determine_target_type(self) -> str:
        """Determine if target is categorical or numerical"""
        if self.data[self.target].dtype in ['object', 'category']:
            return 'classification'
        return 'regression'
    
    def _split_data(self):
        """Split data into reference and current datasets with proper column mapping"""
        try:
            # Calculate split point
            split_idx = int(len(self.data) * self.reference_size)
            
            # Split the data
            self.reference_data = self.data.iloc[:split_idx].copy()
            self.current_data = self.data.iloc[split_idx:].copy()
            
            # Create column mapping using Evidently's ColumnMapping class
            self.column_mapping = ColumnMapping(
                target=self.target,
                numerical_features=self.numerical_cols,
                categorical_features=self.categorical_cols,
                embeddings=None,  # Add if you have embeddings
                task='classification' if self.target_type == 'classification' else 'regression'
            )
            
            print(f"\nData split:")
            print(f"- Reference dataset size: {len(self.reference_data):,}")
            print(f"- Current dataset size: {len(self.current_data):,}")
            
        except Exception as e:
            print(f"Error splitting data: {str(e)}")
            raise

    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables for feature importance calculation"""
        df_encoded = df.copy()
        
        for col in self.categorical_cols + [self.target]:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df_encoded[col] = self.label_encoders[col].transform(df[col].astype(str))
                
        return df_encoded
    
    def analyze_dataset(self) -> Dict:
        """Analyze dataset structure and characteristics"""
        try:
            num_categorical = len(self.categorical_cols)
            num_numerical = len(self.numerical_cols)
            
            print(f"Found {num_categorical} categorical and {num_numerical} numerical columns")
            print(f"Data split: {{'reference_size': {len(self.reference_data)}, 'current_size': {len(self.current_data)}}}")
            
            return {
                'categorical_columns': self.categorical_cols,
                'numerical_columns': self.numerical_cols,
                'total_columns': len(self.data.columns),
                'total_rows': len(self.data),
                'reference_size': len(self.reference_data),
                'current_size': len(self.current_data)
            }
        except Exception as e:
            print(f"Error analyzing dataset: {str(e)}")
            raise

    def set_target(self, target_column: str) -> Dict:
        """Set target variable and analyze it"""
        if target_column not in self.data.columns:
            raise ValueError(f"Column {target_column} not found in dataset")
        
        self.target = target_column
        
        # Remove target from feature lists
        if target_column in self.categorical_cols:
            self.categorical_cols.remove(target_column)
            target_type = 'categorical'
        if target_column in self.numerical_cols:
            self.numerical_cols.remove(target_column)
            target_type = 'numerical'
        
        # Create column mapping for Evidently
        self.column_mapping = ColumnMapping(
            target=target_column,
            numerical_features=self.numerical_cols,
            categorical_features=self.categorical_cols
        )
        
        # Analyze target
        target_report = Report(metrics=[
            TargetDriftPreset(),
        ])
        
        target_report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping
        )
        
        # Get target metrics safely
        try:
            target_metrics = target_report.as_dict()
        except:
            target_metrics = {"error": "Could not extract target metrics"}
        
        # Get target distribution
        target_distribution = {
            'reference': self.reference_data[target_column].value_counts().to_dict(),
            'current': self.current_data[target_column].value_counts().to_dict()
        }
        
        return {
            'target_column': target_column,
            'target_type': target_type,
            'target_metrics': target_metrics,
            'target_distribution': target_distribution,
            'feature_split': {
                'categorical_features': len(self.categorical_cols),
                'numerical_features': len(self.numerical_cols)
            }
        }
    
    def analyze_features(self) -> Dict:
        """Analyze features and their characteristics"""
        try:
            feature_analysis = {}
            
            for column in self.data.columns:
                if column != self.target:
                    feature_info = {
                        'type': 'numerical' if column in self.numerical_cols else 'categorical',
                        'unique_values': len(self.data[column].unique()),
                        'missing_values': self.data[column].isnull().sum(),
                        'cardinality': len(self.data[column].unique()) / len(self.data),
                    }
                    
                    # Add numerical statistics if applicable
                    if column in self.numerical_cols:
                        feature_info.update({
                            'mean': float(self.data[column].mean()),
                            'std': float(self.data[column].std()),
                            'min': float(self.data[column].min()),
                            'max': float(self.data[column].max())
                        })
                    
                    feature_analysis[column] = feature_info
            
            return feature_analysis
            
        except Exception as e:
            print(f"Error analyzing features: {str(e)}")
            raise

    def get_feature_importance(self, methods: List[str] = None) -> Dict:
        """Calculate feature importance using different methods with progress tracking"""
        if methods is None:
            methods = ['random_forest']
        
        print("\nPreparing data...")
        # Encode categorical variables
        data_encoded = self._encode_categorical_variables(self.data)
        
        # Filter out ID columns and other non-meaningful features
        excluded_patterns = ['id', 'ID', 'identifier', 'index']
        features_to_use = [col for col in data_encoded.columns 
                          if not any(pattern.lower() in col.lower() 
                                   for pattern in excluded_patterns)]
        
        # Prepare data
        X = data_encoded[features_to_use].drop(columns=[self.target])
        y = data_encoded[self.target]
        feature_names = X.columns.tolist()
        
        importance_results = {}
        
        # Create progress bar for methods
        with tqdm(total=len(methods), desc="Methods", position=0) as pbar:
            for method in methods:
                try:
                    start_time = time.time()
                    
                    if method == 'random_forest':
                        pbar.set_description(f"Random Forest")
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        
                        with tqdm(total=2, desc="Steps", position=1, leave=False) as inner_pbar:
                            inner_pbar.set_description("Training model")
                            model.fit(X, y)
                            inner_pbar.update(1)
                            
                            inner_pbar.set_description("Calculating importance")
                            importance_values = model.feature_importances_
                            inner_pbar.update(1)
                        
                        importance_results['random_forest'] = {
                            'importance_values': importance_values.tolist(),
                            'feature_names': feature_names,
                            'sorted_features': [x for _, x in sorted(
                                zip(importance_values, feature_names),
                                reverse=True
                            )],
                            'computation_time': f"{time.time() - start_time:.2f} seconds"
                        }
                    
                    elif method == 'permutation':
                        pbar.set_description(f"Permutation")
                        
                        with tqdm(total=3, desc="Steps", position=1, leave=False) as inner_pbar:
                            inner_pbar.set_description("Training model")
                            # Use a simpler model for speed
                            model = RandomForestClassifier(
                                n_estimators=50,  # Reduced from 100
                                max_depth=10,     # Limit tree depth
                                random_state=42
                            )
                            model.fit(X, y)
                            inner_pbar.update(1)
                            
                            inner_pbar.set_description("Computing permutations")
                            # Use parallel processing with timeout
                            with parallel_backend('threading', n_jobs=-1):
                                try:
                                    perm_importance = permutation_importance(
                                        model, X, y,
                                        n_repeats=5,     # Reduced from 10
                                        random_state=42,
                                        n_jobs=-1,       # Use all CPU cores
                                        scoring='accuracy'
                                    )
                                    inner_pbar.update(1)
                                    
                                    inner_pbar.set_description("Processing results")
                                    importance_results['permutation'] = {
                                        'importance_values': perm_importance.importances_mean.tolist(),
                                        'feature_names': feature_names,
                                        'sorted_features': [x for _, x in sorted(
                                            zip(perm_importance.importances_mean, feature_names),
                                            reverse=True
                                        )],
                                        'computation_time': f"{time.time() - start_time:.2f} seconds"
                                    }
                                    inner_pbar.update(1)
                                    
                                except Exception as e:
                                    print(f"\nPermutation importance calculation timed out or failed. Error: {str(e)}")
                                    importance_results['permutation'] = {
                                        'error': 'Computation timed out or failed',
                                        'computation_time': f"{time.time() - start_time:.2f} seconds"
                                    }
                    
                    elif method == 'shap':
                        pbar.set_description(f"SHAP")
                        
                        with tqdm(total=4, desc="Steps", position=1, leave=False) as inner_pbar:
                            inner_pbar.set_description("Training model")
                            # Use a simpler model
                            model = RandomForestClassifier(
                                n_estimators=50,  # Reduced from 100
                                max_depth=10,     # Limit tree depth
                                random_state=42
                            )
                            
                            # Use a subset of data for training
                            sample_size = min(10000, len(X))
                            X_sample = X.sample(n=sample_size, random_state=42)
                            y_sample = y[X_sample.index]
                            model.fit(X_sample, y_sample)
                            inner_pbar.update(1)
                            
                            inner_pbar.set_description("Creating explainer")
                            # Use TreeExplainer with faster parameters
                            explainer = shap.TreeExplainer(
                                model,
                                feature_perturbation='interventional',
                                approximate=True
                            )
                            inner_pbar.update(1)
                            
                            inner_pbar.set_description("Computing SHAP values")
                            # Use a small subset for SHAP values
                            X_explain = X.sample(n=min(1000, len(X)), random_state=42)
                            shap_values = explainer.shap_values(X_explain)
                            inner_pbar.update(1)
                            
                            inner_pbar.set_description("Processing results")
                            if isinstance(shap_values, list):
                                shap_values = np.abs(shap_values).mean(axis=0)
                            shap_importance = np.abs(shap_values).mean(axis=0)
                            
                            importance_results['shap'] = {
                                'importance_values': shap_importance.tolist(),
                                'feature_names': feature_names,
                                'sorted_features': [x for _, x in sorted(
                                    zip(shap_importance, feature_names),
                                    reverse=True
                                )],
                                'computation_time': f"{time.time() - start_time:.2f} seconds",
                                'note': f"Computed using {sample_size} samples for training and 1000 samples for SHAP values"
                            }
                            inner_pbar.update(1)
                    
                    elif method == 'mutual_info':
                        pbar.set_description(f"Mutual Information")
                        with tqdm(total=2, desc="Steps", position=1, leave=False) as inner_pbar:
                            inner_pbar.set_description("Computing mutual information")
                            from sklearn.feature_selection import mutual_info_classif
                            mi = mutual_info_classif(X, y)
                            inner_pbar.update(1)
                            
                            inner_pbar.set_description("Processing results")
                            importance_results['mutual_info'] = {
                                'importance_values': mi.tolist(),
                                'feature_names': feature_names,
                                'sorted_features': [x for _, x in sorted(
                                    zip(mi, feature_names),
                                    reverse=True
                                )],
                                'computation_time': f"{time.time() - start_time:.2f} seconds"
                            }
                            inner_pbar.update(1)
                    
                except Exception as e:
                    importance_results[method] = {
                        'error': str(e),
                        'computation_time': f"{time.time() - start_time:.2f} seconds"
                    }
                
                pbar.update(1)
        
        return importance_results
    
    def test_features(self) -> Dict:
        """Run statistical tests on features"""
        test_suite = TestSuite(tests=[
            # Basic data tests
            TestNumberOfRows(),
            TestColumnsType(),
            TestShareOfMissingValues(),
        ])
        
        test_suite.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping
        )
        
        # Create drift report separately
        drift_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset()
        ])
        
        drift_report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping
        )
        
        try:
            return {
                'test_suite_results': test_suite.as_dict(),
                'drift_analysis': drift_report.as_dict()
            }
        except Exception as e:
            return {"error": f"Could not extract test results: {str(e)}"}
    
    def save_powerbi_format(self, results_dict: Dict, output_dir: str):
        """Save results in Power BI friendly CSV format"""
        rows = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = self.target.lower()
        
        # Extract feature-specific results
        feature_results = results_dict.get('feature_analysis', {})
        importance_results = results_dict.get('feature_importance', {})
        drift_results = results_dict.get('drift_results', {})
        
        for feature in self.data.columns:
            try:
                # Initialize base row with feature metadata
                base_row = {
                    'test_name': test_name,
                    'analysis_timestamp': timestamp,
                    'analysis_date': datetime.now().strftime("%Y-%m-%d"),
                    'analysis_time': datetime.now().strftime("%H:%M:%S"),
                    'feature_name': feature,
                    'feature_type': 'numerical' if feature in self.numerical_cols else 'categorical',
                    'unique_values': len(self.reference_data[feature].unique()),
                    'missing_values': self.reference_data[feature].isnull().sum(),
                    'unique_ratio': len(self.reference_data[feature].unique()) / len(self.reference_data),
                    'is_target': feature == self.target,
                    'target_variable': self.target,
                    'target_type': self.target_type,
                    'total_features': len(self.data.columns),
                    'numerical_features': len(self.numerical_cols),
                    'categorical_features': len(self.categorical_cols),
                    'reference_size': len(self.reference_data),
                    'current_size': len(self.current_data)
                }
                
                # Add drift results
                if feature in drift_results:
                    feature_drift = drift_results[feature]
                    for test_name, test_results in feature_drift.items():
                        if isinstance(test_results, dict):
                            base_row[f'{test_name}_drift_score'] = test_results.get('drift_score')
                            base_row[f'{test_name}_p_value'] = test_results.get('p_value')
                            base_row[f'{test_name}_is_drifted'] = test_results.get('is_drifted', False)
                    
                    # Set overall drift status
                    is_drifted = any(test.get('is_drifted', False) 
                                   for test in feature_drift.values() 
                                   if isinstance(test, dict))
                    base_row['drift_status'] = 'Drifted' if is_drifted else 'No Drift'
                
                # Add importance scores
                if feature in importance_results:
                    base_row['random_forest_importance'] = importance_results[feature].get('random_forest', 0)
                    base_row['random_forest_rank'] = importance_results[feature].get('rank', 0)
                
                rows.append(base_row)
                
            except Exception as e:
                print(f"Error processing {feature}: {str(e)}")
                continue
        
        # Save to CSV
        output_path = f"{output_dir}/feature_analysis_{test_name}_{timestamp}.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"\nPower BI friendly file saved as: {output_path}")
    
    def calculate_drift_scores(self) -> Dict:
        """Calculate drift scores with individual column test selection"""
        drift_results = {}
        total_columns = len(self.data.columns) - 1  # Excluding target
        current_column = 0
        
        print("\nCalculating drift scores column by column...")
        
        for column in self.data.columns:
            if column == self.target:
                continue
                
            current_column += 1
            is_numerical = column in self.numerical_cols
            drift_results[column] = {}
            
            print(f"\nColumn {current_column}/{total_columns}: {column}")
            
            if is_numerical:
                print("Available tests for this numerical column:")
                print("1. Kolmogorov-Smirnov test (ks)")
                print("2. Wasserstein Distance (wasserstein)")
                print("3. Population Stability Index (psi)")
                print("4. Skip this column")
            else:
                print("Available tests for this categorical column:")
                print("1. Chi-square test (chisquare)")
                print("2. Population Stability Index (psi)")
                print("3. Jensen-Shannon Distance (jensenshannon)")
                print("4. Skip this column")
                
            choice = input("Select test (1-4): ").strip()
            
            try:
                if choice != '4':  # Not skipping
                    if is_numerical:
                        if choice == '1':
                            self._run_numerical_test(column, 'ks', drift_results)
                        elif choice == '2':
                            self._run_numerical_test(column, 'wasserstein', drift_results)
                        elif choice == '3':
                            self._run_numerical_test(column, 'psi', drift_results)
                    else:
                        if choice == '1':
                            self._run_categorical_test(column, 'chisquare', drift_results)
                        elif choice == '2':
                            self._run_categorical_test(column, 'psi', drift_results)
                        elif choice == '3':
                            self._run_categorical_test(column, 'jensenshannon', drift_results)
            except Exception as e:
                print(f"Error calculating drift for {column}: {str(e)}")
                
        return drift_results
    
    def _run_numerical_test(self, column: str, test_name: str, results: Dict):
        """Helper method to run numerical drift tests with improved status tracking"""
        try:
            drift_metric = ColumnDriftMetric(
                column_name=column,
                stattest=test_name
            )
            drift_report = Report(metrics=[drift_metric])
            drift_report.run(
                reference_data=self.reference_data,
                current_data=self.current_data,
                column_mapping=self.column_mapping
            )
            result = drift_report.as_dict()['metrics'][0]['result']
            
            # Store test results
            results[column][test_name] = {
                'drift_score': result.get('drift_score', None),
                'p_value': result.get('p_value', None),
                'is_drifted': result.get('drift_detected', False)
            }
            
            # Update drift status
            if results[column][test_name]['is_drifted']:
                results[column]['drift_status'] = 'Drifted'
                results[column]['drift_detection_method'] = test_name
            else:
                results[column]['drift_status'] = 'No Drift'
                
        except Exception as e:
            print(f"Error in {test_name} test for {column}: {str(e)}")
            results[column][test_name] = {
                'drift_score': None,
                'p_value': None,
                'is_drifted': False,
                'error': str(e)
            }

    def _run_categorical_test(self, column: str, test_name: str, results: Dict):
        """Helper method to run categorical drift tests with improved error handling"""
        try:
            # Handle high cardinality columns differently
            unique_ratio = len(self.reference_data[column].unique()) / len(self.reference_data)
            if unique_ratio > 0.1:  # More than 10% unique values
                print(f"Warning: {column} has high cardinality ({unique_ratio:.1%} unique values)")
                print("Using binning for more reliable results...")
                # Implement binning logic here if needed
                
            drift_metric = ColumnDriftMetric(
                column_name=column,
                stattest=test_name
            )
            drift_report = Report(metrics=[drift_metric])
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                drift_report.run(
                    reference_data=self.reference_data,
                    current_data=self.current_data,
                    column_mapping=self.column_mapping
                )
                
            result = drift_report.as_dict()['metrics'][0]['result']
            
            # Store test results
            results[column][test_name] = {
                'drift_score': result.get('drift_score', None),
                'p_value': result.get('p_value', None) if test_name != 'psi' else None,
                'is_drifted': result.get('drift_detected', False)
            }
            
            # Update drift status
            if results[column][test_name]['is_drifted']:
                results[column]['drift_status'] = 'Drifted'
                results[column]['drift_detection_method'] = test_name
            else:
                results[column]['drift_status'] = 'No Drift'
                
        except Exception as e:
            print(f"Error in {test_name} test for {column}: {str(e)}")
            results[column][test_name] = {
                'drift_score': None,
                'p_value': None,
                'is_drifted': False,
                'error': str(e)
            }
    
    def _show_drift_test_menu(self, column_name: str, column_type: str) -> List[str]:
        """Show interactive menu for selecting drift detection methods based on column type"""
        
        numerical_tests = {
            '1': 'wasserstein',      # For numerical data with >1000 objects
            '2': 'ks',               # Kolmogorov-Smirnov test
            '3': 'anderson',         # Anderson-Darling test
            '4': 'cramer_von_mises', # Cramer-Von Mises test
            '5': 'mannw',            # Mann-Whitney U test
            '6': 't_test',           # T-Test
            '7': 'empirical_mmd',    # Empirical MMD
            '8': 'ed',               # Energy distance
            '9': 'es',               # Epps-Singleton test
            '10': 'psi',             # Population Stability Index
            '11': 'kl_div',          # Kullback-Leibler divergence
            '12': 'hellinger',       # Hellinger Distance
            '13': 'all'
        }
        
        categorical_tests = {
            '1': 'chi_square',       # For categorical with >2 labels
            '2': 'fisher_exact',     # Fisher's Exact test
            '3': 'g_test',           # G-test
            '4': 'tvd',              # Total Variation Distance
            '5': 'psi',              # Population Stability Index
            '6': 'kl_div',           # Kullback-Leibler divergence
            '7': 'hellinger',        # Hellinger Distance
            '8': 'jensenshannon',    # For categorical with >1000 objects
            '9': 'all'
        }
        
        while True:
            print(f"\nDrift Detection Methods for {column_name} ({column_type}):")
            
            if column_type == 'numerical':
                print("1.  Wasserstein Distance (distribution shape, >1000 samples)")
                print("2.  Kolmogorov-Smirnov Test (distribution shape)")
                print("3.  Anderson-Darling Test (distribution goodness of fit)")
                print("4.  Cramer-Von Mises Test (distribution goodness of fit)")
                print("5.  Mann-Whitney U Test (distribution location)")
                print("6.  T-Test (mean comparison)")
                print("7.  Empirical MMD (distribution similarity)")
                print("8.  Energy Distance (distribution similarity)")
                print("9.  Epps-Singleton Test (distribution comparison)")
                print("10. Population Stability Index (PSI)")
                print("11. Kullback-Leibler Divergence (distribution similarity)")
                print("12. Hellinger Distance (probability distributions)")
                print("13. All Methods")
                tests = numerical_tests
            else:  # categorical
                print("1. Chi-Square Test (frequency distribution)")
                print("2. Fisher's Exact Test (contingency tables)")
                print("3. G-Test (likelihood ratio)")
                print("4. Total Variation Distance (probability distributions)")
                print("5. Population Stability Index (PSI)")
                print("6. Kullback-Leibler Divergence")
                print("7. Hellinger Distance")
                print("8. Jensen-Shannon Distance (>1000 samples)")
                print("9. All Methods")
                tests = categorical_tests
                
            print(f"{len(tests) + 1}. Skip this column")
            
            choice = input(f"\nSelect method for {column_name} (1-{len(tests) + 1}): ").strip()
            
            if choice == str(len(tests) + 1):
                return []
            
            if choice in tests:
                if tests[choice] == 'all':
                    return list(tests.values())[:-1]  # All except 'all' option
                return [tests[choice]]
            else:
                print("Invalid choice! Please try again.")

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        try:
            # 1. Dataset structure analysis
            print("\n1. Analyzing dataset structure...")
            dataset_analysis = self.analyze_dataset()
            
            # 2. Set target variable
            print("\n2. Setting target variable...")
            print(f"Target variable: {self.target} ({self.target_type})")
            
            # 3. Feature analysis
            print("\n3. Analyzing features...")
            feature_analysis = self.analyze_features()
            print("Feature analysis complete")
            
            # 4. Feature importance
            print("\n4. Calculating feature importance...")
            importance_results = self.calculate_feature_importance()
            
            # 5. Drift analysis
            print("\n5. Calculating drift scores...")
            drift_results = self.calculate_drift_scores()
            
            # 6. Statistical tests
            print("\n6. Running statistical tests...")
            statistical_results = self.run_statistical_tests()
            print("Statistical tests complete")
            
            # Combine all results
            results = {
                'dataset_analysis': dataset_analysis,
                'target_analysis': {'target': self.target, 'type': self.target_type},
                'feature_analysis': feature_analysis,
                'feature_importance': importance_results,
                'drift_results': drift_results,
                'statistical_tests': statistical_results
            }
            
            return results
            
        except Exception as e:
            print(f"Error in analysis pipeline: {str(e)}")
            raise

    def calculate_feature_importance(self):
        """Calculate feature importance using selected methods"""
        try:
            print("\nFeature Importance Methods:")
            print("1. Random Forest (faster, built-in importance)")
            print("2. Permutation Importance (slower, model agnostic)")
            print("3. SHAP Values (slower, more detailed)")
            print("4. Mutual Information (faster, statistical measure)")
            print("5. Done selecting")
            
            selected_methods = []
            while True:
                choice = input("\nSelect a method (1-5): ")
                if choice == '5':
                    break
                elif choice in ['1', '2', '3', '4']:
                    method_name = {
                        '1': 'random_forest',
                        '2': 'permutation',
                        '3': 'shap',
                        '4': 'mutual_info'
                    }[choice]
                    selected_methods.append(method_name)
                    print(f"Selected methods: {selected_methods}")
                else:
                    print("Invalid choice. Please select 1-5.")
                
            if not selected_methods:
                return {}
                
            print(f"\nCalculating importance using: {selected_methods}")
            importance_results = {}
            
            # Prepare data
            print("\nPreparing data...")
            X = self.reference_data.drop(columns=[self.target])
            
            # Handle categorical variables
            categorical_encoders = {}
            X_encoded = X.copy()
            for col in self.categorical_cols:
                if col != self.target:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                    categorical_encoders[col] = le
            
            # Encode target
            y_encoder = LabelEncoder()
            y = y_encoder.fit_transform(self.reference_data[self.target])
            
            # Calculate importance for each method
            for method in tqdm(selected_methods, desc="Random Forest"):
                if method == 'random_forest':
                    # Initialize model
                    if self.target_type == 'classification':
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    else:
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    
                    # Train model and get feature importance
                    with tqdm(total=2, desc="Steps") as pbar:
                        pbar.set_description("Training model")
                        model.fit(X_encoded, y)
                        pbar.update(1)
                        
                        pbar.set_description("Calculating importance")
                        importance_scores = model.feature_importances_
                        pbar.update(1)
                    
                    # Store results
                    for feature, importance in zip(X.columns, importance_scores):
                        if feature not in importance_results:
                            importance_results[feature] = {}
                        importance_results[feature]['random_forest'] = float(importance)
                    
                    # Sort features by importance and store ranks
                    sorted_features = sorted(zip(X.columns, importance_scores), 
                                          key=lambda x: x[1], reverse=True)
                    
                    print("\nTop 10 features using random_forest:")
                    for feature, score in sorted_features[:10]:
                        print(f"- {feature}")
                        if feature not in importance_results:
                            importance_results[feature] = {}
                        importance_results[feature]['rank'] = list(zip(*sorted_features))[0].index(feature) + 1
                
                # Add other methods here if selected
                # elif method == 'permutation':
                # elif method == 'shap':
                # elif method == 'mutual_info':
            
            return importance_results
            
        except Exception as e:
            print(f"Error calculating feature importance: {str(e)}")
            raise

    def run_statistical_tests(self):
        """Run statistical tests on the data"""
        try:
            statistical_results = {}
            
            # 1. Distribution tests
            for column in self.numerical_cols:
                if column != self.target:
                    # Kolmogorov-Smirnov test for distribution comparison
                    ref_data = self.reference_data[column].dropna()
                    curr_data = self.current_data[column].dropna()
                    
                    if len(ref_data) > 0 and len(curr_data) > 0:
                        ks_stat, p_value = stats.ks_2samp(ref_data, curr_data)
                        
                        statistical_results[column] = {
                            'ks_test': {
                                'statistic': float(ks_stat),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05
                            }
                        }
            
            # 2. Correlation analysis
            ref_corr = self.reference_data[self.numerical_cols].corr()
            curr_corr = self.current_data[self.numerical_cols].corr()
            
            correlation_changes = {}
            for col1 in self.numerical_cols:
                for col2 in self.numerical_cols:
                    if col1 < col2:  # Only look at unique pairs
                        ref_val = ref_corr.loc[col1, col2]
                        curr_val = curr_corr.loc[col1, col2]
                        change = abs(ref_val - curr_val)
                        
                        if change > 0.1:  # Report significant correlation changes
                            correlation_changes[f"{col1}_vs_{col2}"] = {
                                'reference_correlation': float(ref_val),
                                'current_correlation': float(curr_val),
                                'absolute_change': float(change)
                            }
            
            # 3. Basic statistics comparison
            for column in self.numerical_cols:
                if column not in statistical_results:
                    statistical_results[column] = {}
                
                ref_stats = self.reference_data[column].describe()
                curr_stats = self.current_data[column].describe()
                
                statistical_results[column]['basic_stats'] = {
                    'reference': {
                        'mean': float(ref_stats['mean']),
                        'std': float(ref_stats['std']),
                        'min': float(ref_stats['min']),
                        'max': float(ref_stats['max']),
                        'median': float(ref_stats['50%'])
                    },
                    'current': {
                        'mean': float(curr_stats['mean']),
                        'std': float(curr_stats['std']),
                        'min': float(curr_stats['min']),
                        'max': float(curr_stats['max']),
                        'median': float(curr_stats['50%'])
                    }
                }
            
            # 4. Categorical variables distribution comparison
            for column in self.categorical_cols:
                if column != self.target:
                    ref_counts = self.reference_data[column].value_counts()
                    curr_counts = self.current_data[column].value_counts()
                    
                    # Align categories and normalize
                    all_categories = sorted(set(ref_counts.index) | set(curr_counts.index))
                    ref_counts = ref_counts.reindex(all_categories).fillna(0)
                    curr_counts = curr_counts.reindex(all_categories).fillna(0)
                    
                    # Convert to proportions
                    ref_props = ref_counts / ref_counts.sum()
                    curr_props = curr_counts / curr_counts.sum()
                    
                    # Calculate chi-square statistic manually
                    expected = ref_props * len(self.current_data)
                    observed = curr_counts
                    
                    valid_mask = expected > 0  # Only use categories present in reference
                    if valid_mask.sum() > 0:
                        chi2_stat = np.sum(((observed[valid_mask] - expected[valid_mask]) ** 2) / expected[valid_mask])
                        df = valid_mask.sum() - 1
                        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
                        
                        statistical_results[column] = {
                            'distribution_test': {
                                'statistic': float(chi2_stat),
                                'p_value': float(p_value),
                                'degrees_of_freedom': int(df),
                                'significant': p_value < 0.05
                            }
                        }
            
            return {
                'numerical_tests': statistical_results,
                'correlation_changes': correlation_changes
            }
            
        except Exception as e:
            print(f"Error running statistical tests: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Get data path from user
        data_path = input("Enter path to your data file: ")
        data = pd.read_csv(data_path)
        
        # Show columns and allow dropping
        while True:
            print("\nAvailable columns:")
            for i, col in enumerate(data.columns, 1):
                print(f"{i}. {col}")
            
            print("\nOptions:")
            print("1. Drop columns")
            print("2. Continue with analysis")
            choice = input("\nSelect option (1-2): ")
            
            if choice == "1":
                drop_cols = input("\nEnter column numbers to drop (comma-separated, e.g., 1,2,3): ")
                try:
                    # Convert input to column indices
                    drop_indices = [int(x.strip()) - 1 for x in drop_cols.split(",")]
                    # Get column names to drop
                    cols_to_drop = [data.columns[i] for i in drop_indices]
                    # Drop columns
                    data = data.drop(columns=cols_to_drop)
                    print(f"\nDropped columns: {', '.join(cols_to_drop)}")
                except Exception as e:
                    print(f"Error dropping columns: {str(e)}")
            else:
                break
        
        # Get target variable from user
        print("\nSelect target variable:")
        for i, col in enumerate(data.columns, 1):
            print(f"{i}. {col}")
        target_idx = int(input("\nEnter number: ")) - 1
        target = data.columns[target_idx]
        
        # Initialize assistant
        assistant = EvidentlyAssistant(
            data=data,
            target=target,
            reference_size=0.7
        )
        
        # Run analysis
        results = assistant.run_analysis()
        
        # Save results
        output_dir = "evidently_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON results using NumpyEncoder
        with open(f"{output_dir}/analysis_results.json", 'w') as f:
            json.dump(results, f, cls=NumpyEncoder, indent=4)
        print(f"\nResults saved to {output_dir}/analysis_results.json")
        
        # Save Power BI format
        assistant.save_powerbi_format(results, output_dir)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
