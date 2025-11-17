# File: tests/test_analysis.py

import pytest
import numpy as np
import pandas as pd
from src.analysis import ABTestAnalyzer

class TestABTestAnalyzer:
    """
    Unit tests for A/B test analysis functions.

    Why testing matters: Ensures statistical calculations are correct
    One small bug in significance testing = millions in bad decisions
    """

    def setup_method(self):
        """Run before each test"""
        self.analyzer = ABTestAnalyzer(alpha=0.05)

    def test_two_proportion_z_test_significant_difference(self):
        """
        Test that z-test correctly identifies significant difference.
        """
        # Create scenario where difference is clearly significant
        # Control: 1000/10000 = 10%
        # Treatment: 1500/10000 = 15% (50% relative lift)

        z_stat, p_value = self.analyzer._two_proportion_z_test(
            x1=1000, n1=10000,
            x2=1500, n2=10000
        )

        # Assertions
        assert p_value < 0.05, "Should detect significant difference"
        assert z_stat > 0, "Z-statistic should be positive for positive lift"

    def test_two_proportion_z_test_no_difference(self):
        """
        Test that z-test correctly identifies no difference.
        """
        # Both groups have same conversion rate (15%)
        z_stat, p_value = self.analyzer._two_proportion_z_test(
            x1=1500, n1=10000,
            x2=1500, n2=10000
        )

        assert p_value > 0.05, "Should not detect difference when rates are equal"
        assert abs(z_stat) < 0.1, "Z-statistic should be near zero"

    def test_cuped_reduces_variance(self):
        """
        Test that CUPED actually reduces variance.
        """
        # Create synthetic data with correlated pre-experiment metric
        np.random.seed(42)
        n = 1000

        data = pd.DataFrame({
            'user_id': range(n),
            'variant': ['control'] * (n//2) + ['treatment'] * (n//2),
            'pre_metric': np.random.normal(10, 3, n),
        })

        # Post-metric is correlated with pre-metric
        data['metric'] = data['pre_metric'] * 0.8 + np.random.normal(0, 2, n)
        data.loc[data['variant'] == 'treatment', 'metric'] += 1.5  # Treatment effect

        results = self.analyzer.analyze_with_cuped(data, 'pre_metric')

        # CUPED should reduce variance
        original_var = np.var(data['metric'])
        cuped_var = np.var(data['cuped_metric'])

        assert cuped_var < original_var, "CUPED should reduce variance"
        assert results['cuped_analysis']['is_significant'], "Should detect effect with CUPED"

    def test_bayesian_probabilities_sum_to_one(self):
        """
        Test that Bayesian probabilities are valid.
        """
        results = self.analyzer.bayesian_ab_test(
            control_conversions=1500,
            control_total=10000,
            treatment_conversions=1750,
            treatment_total=10000
        )

        prob_b_beats_a = results['probability_b_beats_a']

        # Probability should be between 0 and 1
        assert 0 <= prob_b_beats_a <= 1, "Probability must be between 0 and 1"

        # With this data, treatment should very likely be better
        assert prob_b_beats_a > 0.9, "Should be very confident treatment is better"

    def test_sample_size_calculation(self):
        """
        Test that sample size calculator gives reasonable results.
        """
        from src.experiment_design import ExperimentDesigner

        designer = ExperimentDesigner()

        sample_size = designer.calculate_sample_size(
            baseline_rate=0.15,
            minimum_detectable_effect=0.02,
            alpha=0.05,
            power=0.80
        )

        # Sample size should be positive and reasonable
        assert sample_size > 0, "Sample size must be positive"
        assert sample_size < 100000, "Sample size seems unreasonably large"

        # Larger MDE should require smaller sample
        smaller_sample = designer.calculate_sample_size(
            baseline_rate=0.15,
            minimum_detectable_effect=0.05,  # Larger effect
            alpha=0.05,
            power=0.80
        )

        assert smaller_sample < sample_size, "Larger MDE should need smaller sample"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
