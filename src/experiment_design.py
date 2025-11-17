# File: src/experiment_design.py

import numpy as np
from scipy import stats
import math

class ExperimentDesigner:
    """
    Tools for planning experiments BEFORE you run them.

    Why this matters: Running experiments costs money and time.
    Bad planning = unreliable results = bad decisions.
    """

    def calculate_sample_size(self,
                             baseline_rate,
                             minimum_detectable_effect,
                             alpha=0.05,
                             power=0.80):
        """
        Calculates how many users you need per variant.

        Parameters:
        -----------
        baseline_rate: Current conversion rate (e.g., 0.15 = 15%)
        minimum_detectable_effect: Smallest change you care about (e.g., 0.02 = 2%)
        alpha: Significance level (probability of false positive, default 5%)
        power: Probability of detecting real effect (default 80%)

        Returns:
        --------
        Required sample size per variant

        Example:
        --------
        If 15% of users convert now, and you want to detect a 2% improvement
        (to 17%), how many users do you need?

        >>> designer = ExperimentDesigner()
        >>> designer.calculate_sample_size(0.15, 0.02)
        12850  # You need ~12,850 users per variant
        """

        # What these terms mean:
        # baseline_rate: Your current performance
        # minimum_detectable_effect: How much better needs to be worth it?
        # alpha: Chance you'll think there's a difference when there isn't (false alarm)
        # power: Chance you'll catch a real difference (sensitivity)

        # Calculate the "better" rate you're testing for
        effect_size = baseline_rate + minimum_detectable_effect

        # This formula comes from statistical theory (trust me, it works!)
        # It balances:
        # - How different the two rates are (bigger difference = fewer users needed)
        # - How confident you want to be (higher confidence = more users needed)
        # - Statistical power (higher power = more users needed)

        z_alpha = stats.norm.ppf(1 - alpha/2)  # Critical value for significance
        z_beta = stats.norm.ppf(power)          # Critical value for power

        # Pooled standard deviation
        pooled_prob = (baseline_rate + effect_size) / 2
        pooled_std = math.sqrt(2 * pooled_prob * (1 - pooled_prob))

        # Sample size formula
        n = ((z_alpha + z_beta) * pooled_std / minimum_detectable_effect) ** 2

        return math.ceil(n)  # Round up to be safe

    def calculate_test_duration(self,
                                sample_size_per_variant,
                                daily_traffic,
                                n_variants=2):
        """
        How many days should you run the experiment?

        Parameters:
        -----------
        sample_size_per_variant: Result from calculate_sample_size()
        daily_traffic: How many new users you get per day
        n_variants: How many groups (e.g., 2 for A/B, 3 for A/B/C)

        Returns:
        --------
        Number of days to run the experiment
        """
        total_users_needed = sample_size_per_variant * n_variants
        days = math.ceil(total_users_needed / daily_traffic)

        return days

    def create_experiment_plan(self,
                              experiment_name,
                              baseline_rate,
                              mde,
                              daily_traffic,
                              n_variants=2):
        """
        Creates a complete pre-experiment summary.

        This is what you'd send to your team before launching.
        """
        sample_size = self.calculate_sample_size(baseline_rate, mde)
        duration = self.calculate_test_duration(sample_size, daily_traffic, n_variants)

        plan = {
            'experiment_name': experiment_name,
            'baseline_conversion_rate': f"{baseline_rate:.1%}",
            'minimum_detectable_effect': f"{mde:.1%}",
            'expected_new_rate': f"{baseline_rate + mde:.1%}",
            'sample_size_per_variant': f"{sample_size:,}",
            'total_users_needed': f"{sample_size * n_variants:,}",
            'daily_traffic': f"{daily_traffic:,}",
            'estimated_duration_days': duration,
            'estimated_duration_weeks': round(duration / 7, 1),
            'significance_level': '5% (α = 0.05)',
            'statistical_power': '80%',
            'number_of_variants': n_variants
        }

        return plan

    def print_experiment_plan(self, plan):
        """
        Pretty-prints the experiment plan.
        """
        print("="*60)
        print(f"EXPERIMENT PLAN: {plan['experiment_name']}")
        print("="*60)
        print(f"\n📊 Current Performance:")
        print(f"   Baseline Conversion Rate: {plan['baseline_conversion_rate']}")
        print(f"\n🎯 What We're Testing For:")
        print(f"   Minimum Detectable Effect: {plan['minimum_detectable_effect']}")
        print(f"   Expected New Rate: {plan['expected_new_rate']}")
        print(f"\n👥 Sample Size Requirements:")
        print(f"   Per Variant: {plan['sample_size_per_variant']} users")
        print(f"   Total Needed: {plan['total_users_needed']} users")
        print(f"\n⏱️ Timeline:")
        print(f"   Daily Traffic: {plan['daily_traffic']} users/day")
        print(f"   Duration: {plan['estimated_duration_days']} days ({plan['estimated_duration_weeks']} weeks)")
        print(f"\n📈 Statistical Parameters:")
        print(f"   Significance Level: {plan['significance_level']}")
        print(f"   Statistical Power: {plan['statistical_power']}")
        print(f"   Number of Variants: {plan['number_of_variants']}")
        print("="*60)

# EXAMPLE USAGE
if __name__ == "__main__":
    designer = ExperimentDesigner()

    # Scenario: We want to test if adding PayPal increases conversions
    # Current rate: 15%
    # We want to detect at least a 2% improvement (to 17%)
    # We get 5,000 new visitors per day

    plan = designer.create_experiment_plan(
        experiment_name="PayPal Payment Option Test",
        baseline_rate=0.15,
        mde=0.02,
        daily_traffic=5000,
        n_variants=2
    )

    designer.print_experiment_plan(plan)
