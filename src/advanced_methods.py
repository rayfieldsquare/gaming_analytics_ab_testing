# File: src/advanced_methods.py (new file!)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MultiArmedBandit:
    """
    Implements Thompson Sampling for multi-armed bandit optimization.

    Use this when you want to:
    1. Test multiple variants (3+)
    2. Maximize performance DURING the test (not just after)
    3. Dynamically allocate traffic to winners
    """

    def __init__(self, variant_names, true_conversion_rates=None):
        """
        Parameters:
        -----------
        variant_names: List of variant names ['A', 'B', 'C']
        true_conversion_rates: Actual rates (for simulation only)
        """
        self.variant_names = variant_names
        self.n_variants = len(variant_names)
        self.true_rates = true_conversion_rates

        # Track results for each variant
        # We use Beta distribution to model uncertainty
        self.alpha = np.ones(self.n_variants)  # Successes + 1
        self.beta = np.ones(self.n_variants)   # Failures + 1

        # Track history
        self.history = []

    def select_variant(self):
        """
        Thompson Sampling: Select variant based on probability of being best.

        How it works:
        1. For each variant, sample from its posterior distribution
        2. Pick whichever variant has the highest sampled value
        3. Over time, this automatically shifts traffic to winners
        """

        # Sample from each variant's posterior distribution
        sampled_values = [
            np.random.beta(self.alpha[i], self.beta[i])
            for i in range(self.n_variants)
        ]

        # Pick the variant with highest sampled value
        selected_idx = np.argmax(sampled_values)

        return selected_idx, self.variant_names[selected_idx]

    def update(self, variant_idx, converted):
        """
        Update beliefs after observing a result.

        Parameters:
        -----------
        variant_idx: Which variant was shown
        converted: Did the user convert? (1 = yes, 0 = no)
        """

        if converted:
            self.alpha[variant_idx] += 1  # Success!
        else:
            self.beta[variant_idx] += 1   # Failure

    def simulate_experiment(self, n_trials=10000):
        """
        Simulates running a multi-armed bandit experiment.

        Parameters:
        -----------
        n_trials: Number of users to show variants to

        Returns:
        --------
        Results dataframe with trial-by-trial data
        """

        if self.true_rates is None:
            raise ValueError("Need true_conversion_rates for simulation")

        results = []

        for trial in range(n_trials):
            # Select which variant to show
            variant_idx, variant_name = self.select_variant()

            # Simulate user conversion (based on true rate)
            converted = np.random.random() < self.true_rates[variant_idx]

            # Update beliefs
            self.update(variant_idx, converted)

            # Record what happened
            results.append({
                'trial': trial,
                'variant_idx': variant_idx,
                'variant_name': variant_name,
                'converted': converted,
                'estimated_rate': self.alpha[variant_idx] / (self.alpha[variant_idx] + self.beta[variant_idx])
            })

            # Track traffic distribution every 100 trials
            if trial % 100 == 0:
                self.history.append({
                    'trial': trial,
                    'traffic_distribution': self.get_traffic_distribution(),
                    'estimated_rates': self.get_estimated_rates()
                })

        return pd.DataFrame(results)

    def get_traffic_distribution(self):
        """
        Returns current probability of selecting each variant.
        """
        # Run 10,000 selections to estimate distribution
        samples = [self.select_variant()[0] for _ in range(10000)]
        counts = np.bincount(samples, minlength=self.n_variants)
        return counts / counts.sum()

    def get_estimated_rates(self):
        """
        Returns current estimated conversion rate for each variant.
        """
        return self.alpha / (self.alpha + self.beta)

    def print_final_results(self, results_df):
        """
        Prints summary of bandit experiment.
        """
        print("="*70)
        print("MULTI-ARMED BANDIT RESULTS")
        print("="*70)

        print("\n🎰 TRUE CONVERSION RATES (Ground Truth):")
        for i, name in enumerate(self.variant_names):
            print(f"   {name}: {self.true_rates[i]:.2%}")

        print("\n📊 FINAL ESTIMATED RATES:")
        estimated = self.get_estimated_rates()
        for i, name in enumerate(self.variant_names):
            print(f"   {name}: {estimated[i]:.2%}")

        print("\n🚦 TRAFFIC DISTRIBUTION:")
        # Calculate actual traffic sent to each variant
        traffic = results_df['variant_name'].value_counts(normalize=True).sort_index()
        for name in self.variant_names:
            pct = traffic.get(name, 0)
            print(f"   {name}: {pct:.1%}")

        print("\n💰 PERFORMANCE METRICS:")
        # Calculate total conversions
        total_conversions = results_df['converted'].sum()
        total_trials = len(results_df)
        actual_rate = total_conversions / total_trials

        # Compare to equal allocation
        equal_allocation_rate = np.mean(self.true_rates)

        improvement = (actual_rate - equal_allocation_rate) / equal_allocation_rate

        print(f"   Total Conversions: {total_conversions:,} / {total_trials:,}")
        print(f"   Realized Conversion Rate: {actual_rate:.2%}")
        print(f"   Equal Allocation Would Give: {equal_allocation_rate:.2%}")
        print(f"   Improvement: {improvement:+.1%}")

        print("\n🏆 WINNER:")
        best_idx = np.argmax(estimated)
        print(f"   Best Variant: {self.variant_names[best_idx]}")
        print(f"   Estimated Rate: {estimated[best_idx]:.2%}")
        print(f"   True Rate: {self.true_rates[best_idx]:.2%}")

        print("="*70)

    def plot_traffic_evolution(self, save_path=None):
        """
        Visualizes how traffic shifted over time.
        """

        if not self.history:
            print("No history to plot. Run simulate_experiment() first.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Extract data from history
        trials = [h['trial'] for h in self.history]
        traffic_data = np.array([h['traffic_distribution'] for h in self.history])

        # Plot 1: Traffic allocation over time
        for i, name in enumerate(self.variant_names):
            ax1.plot(trials, traffic_data[:, i] * 100,
                    label=f'{name} (True: {self.true_rates[i]:.1%})',
                    linewidth=2, marker='o', markersize=3)

        ax1.set_xlabel('Trial Number', fontsize=12)
        ax1.set_ylabel('Traffic Allocation (%)', fontsize=12)
        ax1.set_title('Multi-Armed Bandit: Traffic Allocation Over Time',
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=100/self.n_variants, color='red', linestyle='--',
                   label='Equal Split', alpha=0.5)

        # Plot 2: Estimated rates converging to truth
        rates_data = np.array([h['estimated_rates'] for h in self.history])

        for i, name in enumerate(self.variant_names):
            ax2.plot(trials, rates_data[:, i] * 100,
                    label=f'{name} Estimated',
                    linewidth=2, marker='o', markersize=3)
            ax2.axhline(y=self.true_rates[i] * 100,
                       linestyle='--', alpha=0.7,
                       label=f'{name} True Rate')

        ax2.set_xlabel('Trial Number', fontsize=12)
        ax2.set_ylabel('Conversion Rate (%)', fontsize=12)
        ax2.set_title('Estimated vs True Conversion Rates',
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

# EXAMPLE USAGE
if __name__ == "__main__":
    # Scenario: Testing 3 pricing tiers for a game subscription
    # Variant A: $9.99/mo (converts at 18%)
    # Variant B: $12.99/mo (converts at 15%)
    # Variant C: $14.99/mo (converts at 12%)

    print("🎮 Simulating Multi-Armed Bandit Pricing Test\n")

    bandit = MultiArmedBandit(
        variant_names=['$9.99', '$12.99', '$14.99'],
        true_conversion_rates=[0.18, 0.15, 0.12]
    )

    # Run experiment
    results = bandit.simulate_experiment(n_trials=10000)

    # Print results
    bandit.print_final_results(results)

    # Visualize
    bandit.plot_traffic_evolution(
        save_path='data/processed/bandit_traffic_evolution.png'
    )

    # COMPARISON: What if we did traditional A/B test?
    print("\n" + "="*70)
    print("COMPARISON: Traditional A/B Test (Equal Split)")
    print("="*70)
    print("\nWith equal 33/33/33 split for 10,000 users:")
    print(f"   Expected conversions: {np.mean([0.18, 0.15, 0.12]) * 10000:.0f}")
    print(f"\nWith bandit (dynamic allocation):")
    actual_conversions = results['converted'].sum()
    print(f"   Actual conversions: {actual_conversions}")
    print(f"   Gain: {actual_conversions - np.mean([0.18, 0.15, 0.12]) * 10000:.0f} extra conversions!")
    print("="*70)
