"""
Data drift detection module.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


class DriftDetector:
    """Detect data drift using statistical tests."""
    
    def __init__(self, reference_data: np.ndarray, feature_names: List[str] = None):
        """
        Initialize with reference (training) data.
        
        Args:
            reference_data: Reference dataset (training data)
            feature_names: Names of features
        """
        self.reference_data = reference_data
        self.feature_names = feature_names or [f"feature_{i}" for i in range(reference_data.shape[1])]
        
        self.reference_stats = self._compute_stats(reference_data)
    
    def _compute_stats(self, data: np.ndarray) -> Dict:
        """Compute statistics for a dataset."""
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'median': np.median(data, axis=0),
        }
    
    def detect_drift_ks(self, new_data: np.ndarray, threshold: float = 0.05) -> Dict:
        """
        Detect drift using Kolmogorov-Smirnov test.
        
        Args:
            new_data: New data to check for drift
            threshold: P-value threshold for drift detection
            
        Returns:
            Dictionary with drift results per feature
        """
        results = {}
        
        for i, feature_name in enumerate(self.feature_names):
            statistic, p_value = stats.ks_2samp(
                self.reference_data[:, i],
                new_data[:, i]
            )
            
            results[feature_name] = {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'drift_detected': p_value < threshold
            }
        
        return results
    
    def detect_drift_psi(self, new_data: np.ndarray, bins: int = 10, threshold: float = 0.2) -> Dict:
        """
        Detect drift using Population Stability Index (PSI).
        
        Args:
            new_data: New data to check
            bins: Number of bins for discretization
            threshold: PSI threshold (>0.1 moderate, >0.2 significant)
            
        Returns:
            Dictionary with PSI values per feature
        """
        results = {}
        
        for i, feature_name in enumerate(self.feature_names):
            psi = self._calculate_psi(
                self.reference_data[:, i],
                new_data[:, i],
                bins
            )
            
            results[feature_name] = {
                'psi': float(psi),
                'drift_detected': psi > threshold,
                'severity': 'none' if psi < 0.1 else 'moderate' if psi < 0.2 else 'significant'
            }
        
        return results
    
    def _calculate_psi(self, reference: np.ndarray, new: np.ndarray, bins: int) -> float:
        """Calculate PSI between two distributions."""
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints[-1] = breakpoints[-1] + 1e-6
        
        ref_counts, _ = np.histogram(reference, bins=breakpoints)
        new_counts, _ = np.histogram(new, bins=breakpoints)
        
        ref_percents = ref_counts / len(reference) + 1e-6
        new_percents = new_counts / len(new) + 1e-6
        
        psi = np.sum((new_percents - ref_percents) * np.log(new_percents / ref_percents))
        
        return psi
    
    def get_summary(self, new_data: np.ndarray) -> Dict:
        """Get a summary of drift detection."""
        ks_results = self.detect_drift_ks(new_data)
        psi_results = self.detect_drift_psi(new_data)
        
        drifted_features_ks = [f for f, r in ks_results.items() if r['drift_detected']]
        drifted_features_psi = [f for f, r in psi_results.items() if r['drift_detected']]
        
        return {
            'ks_test': ks_results,
            'psi': psi_results,
            'summary': {
                'total_features': len(self.feature_names),
                'drifted_ks': len(drifted_features_ks),
                'drifted_psi': len(drifted_features_psi),
                'drifted_features_ks': drifted_features_ks,
                'drifted_features_psi': drifted_features_psi,
                'requires_retraining': len(drifted_features_psi) > len(self.feature_names) // 2
            }
        }


def main():
    """Demo drift detection."""
    print("="*60)
    print("DRIFT DETECTION DEMO")
    print("="*60)
    
    np.random.seed(42)
    reference = np.random.randn(1000, 4)
    
    no_drift = np.random.randn(500, 4)
    
    with_drift = np.random.randn(500, 4)
    with_drift[:, 0] += 1.0
    with_drift[:, 2] *= 2.0
    
    detector = DriftDetector(reference, ['f1', 'f2', 'f3', 'f4'])
    
    print("\n1. Testing data WITHOUT drift:")
    summary = detector.get_summary(no_drift)
    print(f"   Drifted features (KS): {summary['summary']['drifted_features_ks']}")
    print(f"   Drifted features (PSI): {summary['summary']['drifted_features_psi']}")
    
    print("\n2. Testing data WITH drift:")
    summary = detector.get_summary(with_drift)
    print(f"   Drifted features (KS): {summary['summary']['drifted_features_ks']}")
    print(f"   Drifted features (PSI): {summary['summary']['drifted_features_psi']}")
    print(f"   Requires retraining: {summary['summary']['requires_retraining']}")


if __name__ == "__main__":
    main()
