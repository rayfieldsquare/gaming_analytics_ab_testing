# File: src/analysis.py

import numpy as np
import pandas as pd
from scipy import stats
import warnings

class ABTestAnalyzer:
    """
    Analyzes A/B test results and determines statistical significance.

    This is where the magic happens - turning data into decisions.
    """

    def __init__(self, alpha=0.05):
        """
        alpha: Significance threshold (default 5% = 0.05)
        """
        self.alpha = alpha

    def analyze_conversion_test(self, control_data, treatment_data):
        """
        Compares conversion rates between two groups.

        Parameters:
        -----------
        control_data: DataFrame with 'converted' column (1=yes, 0=no)
        treatment_data: DataFrame with 'converted' column

        Returns:
        --------
        Dictionary with all test results
        """

        # Calculate summary stats for each group
        control_converted = control_data['converted'].sum()
        control_total = len(control_data)
        control_rate = control_converted / control_total

        treatment_converted = treatment_data['converted'].sum()
        treatment_total = len(treatment_data)
        treatment_rate = treatment_converted / treatment_total

        # Calculate the difference
        absolute_lift = treatment_rate - control_rate
        relative_lift = (treatment_rate - control_rate) / control_rate

        # Perform two-proportion z-test
        # This tests: "Is the difference real or just random luck?"
        z_stat, p_value = self._two_proportion_z_test(
            control_converted, control_total,
            treatment_converted, treatment_total
        )

        # Calculate confidence interval for the difference
        ci_lower, ci_upper = self._calculate_confidence_interval(
            control_rate, treatment_rate,
            control_total, treatment_total
        )

        # Make a decision
        is_significant = p_value < self.alpha

        # Package everything up
        results = {
            'control': {
                'conversions': control_converted,
                'total': control_total,
                'rate': control_rate,
                'rate_pct': f"{control_rate:.2%}"
            },
            'treatment': {
                'conversions': treatment_converted,
                'total': treatment_total,
                'rate': treatment_rate,
                'rate_pct': f"{treatment_rate:.2%}"
            },
            'lift': {
                'absolute': absolute_lift,
                'absolute_pct': f"{absolute_lift:.2%}",
                'relative': relative_lift,
                'relative_pct': f"{relative_lift:.1%}"
            },
            'statistics': {
                'z_statistic': z_stat,
                'p_value': p_value,
                'confidence_interval_95': (ci_lower, ci_upper),
                'confidence_interval_95_pct': f"[{ci_lower:.2%}, {ci_upper:.2%}]",
                'is_significant': is_significant,
                'significance_level': self.alpha
            },
            'recommendation': self._make_recommendation(
                is_significant, absolute_lift, relative_lift
            )
        }

        return results

    def _two_proportion_z_test(self, x1, n1, x2, n2):
        """
        Performs two-proportion z-test.

        This is the standard test for comparing conversion rates.

        Math explained:
        - We have two groups with different conversion rates
        - Question: Could this difference happen by random chance?
        - Answer: Calculate a z-statistic and p-value
        - p-value < 0.05 means "very unlikely to be random chance"
        """

        # Proportions
        p1 = x1 / n1
        p2 = x2 / n2

        # Pooled proportion (weighted average)
        p_pool = (x1 + x2) / (n1 + n2)

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

        # Z-statistic
        z = (p2 - p1) / se

        # P-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return z, p_value

    def _calculate_confidence_interval(self, p1, p2, n1, n2, confidence=0.95):
        """
        Calculates confidence interval for difference in proportions.

        Interpretation: We're 95% confident the true difference is in this range.
        """

        diff = p2 - p1

        # Standard error of the difference
        se = np.sqrt((p1 * (1-p1) / n1) + (p2 * (1-p2) / n2))

        # Critical value (1.96 for 95% confidence)
        z_critical = stats.norm.ppf(1 - (1-confidence)/2)

        # Confidence interval
        ci_lower = diff - z_critical * se
        ci_upper = diff + z_critical * se

        return ci_lower, ci_upper

    def _make_recommendation(self, is_significant, absolute_lift, relative_lift):
        """
        Translates stats into actionable business recommendation.
        """

        if not is_significant:
            return {
                'action': 'DO NOT SHIP',
                'reason': 'Difference is not statistically significant',
                'confidence': 'LOW',
                'explanation': 'The observed difference could easily be due to random chance. Continue running the test or move on.'
            }

        if absolute_lift > 0:
            if relative_lift > 0.10:  # >10% improvement
                confidence = 'VERY HIGH'
                reason = f'Strong positive lift ({relative_lift:.1%})'
            elif relative_lift > 0.05:  # 5-10% improvement
                confidence = 'HIGH'
                reason = f'Moderate positive lift ({relative_lift:.1%})'
            else:  # <5% improvement
                confidence = 'MEDIUM'
                reason = f'Small but significant lift ({relative_lift:.1%})'

            return {
                'action': 'SHIP IT! 🚀',
                'reason': reason,
                'confidence': confidence,
                'explanation': f'Treatment performs better with statistical significance. Expected impact: {relative_lift:.1%} improvement.'
            }
        else:
            return {
                'action': 'DO NOT SHIP',
                'reason': f'Negative impact ({relative_lift:.1%} worse)',
                'confidence': 'HIGH',
                'explanation': 'Treatment performs significantly worse than control. Reject this variant.'
            }

    def print_results(self, results):
        """
        Pretty-prints analysis results for stakeholders.
        """
        print("="*70)
        print("A/B TEST RESULTS")
        print("="*70)

        print("\n📊 CONVERSION RATES:")
        print(f"   Control:   {results['control']['rate_pct']} ({results['control']['conversions']:,}/{results['control']['total']:,})")
        print(f"   Treatment: {results['treatment']['rate_pct']} ({results['treatment']['conversions']:,}/{results['treatment']['total']:,})")

        print("\n📈 LIFT:")
        print(f"   Absolute: {results['lift']['absolute_pct']}")
        print(f"   Relative: {results['lift']['relative_pct']}")

        print("\n🔬 STATISTICAL ANALYSIS:")
        print(f"   Z-Statistic: {results['statistics']['z_statistic']:.4f}")
        print(f"   P-Value: {results['statistics']['p_value']:.6f}")
        print(f"   95% CI: {results['statistics']['confidence_interval_95_pct']}")
        print(f"   Significant? {'✅ YES' if results['statistics']['is_significant'] else '❌ NO'}")

        print("\n💡 RECOMMENDATION:")
        rec = results['recommendation']
        print(f"   Action: {rec['action']}")
        print(f"   Reason: {rec['reason']}")
        print(f"   Confidence: {rec['confidence']}")
        print(f"   Explanation: {rec['explanation']}")

        print("="*70)

    def plot_results(self, results):
        """
        Plots results for stakeholders.
        """
        # generate code to plot results from analysis to a chart here
        

    ################################# CUPED analysis functions ###########################################
    def analyze_with_cuped(self, experiment_data, pre_experiment_metric):
        """
        Uses CUPED to reduce variance and increase test sensitivity.

        Parameters:
        -----------
        experiment_data: DataFrame with columns ['user_id', 'variant', 'metric']
        pre_experiment_metric: Name of column with pre-experiment data

        Returns:
        --------
        Results with variance-reduced estimates

        Example:
        --------
        You're testing a new game tutorial. Before the experiment, you know
        how many games each user played in the past week. CUPED uses this
        to adjust for pre-existing differences.
        """

        # Step 1: Calculate the covariance between pre- and post-metrics
        # This tells us "how predictive is past behavior?"
        covariance = np.cov(
            experiment_data[pre_experiment_metric],
            experiment_data['metric']
        )[0, 1]

        variance_pre = np.var(experiment_data[pre_experiment_metric])

        # Theta is the "adjustment factor"
        # Higher theta = stronger relationship between before & after
        theta = covariance / variance_pre

        # Step 2: Create CUPED-adjusted metric
        # Formula: adjusted_metric = actual_metric - theta * (pre_metric - mean_pre_metric)
        mean_pre = experiment_data[pre_experiment_metric].mean()

        experiment_data['cuped_metric'] = (
            experiment_data['metric'] -
            theta * (experiment_data[pre_experiment_metric] - mean_pre)
        )

        # Step 3: Now analyze using the adjusted metric (less noisy!)
        control = experiment_data[experiment_data['variant'] == 'control']
        treatment = experiment_data[experiment_data['variant'] == 'treatment']

        # Regular analysis (without CUPED)
        regular_result = self._t_test(
            control['metric'],
            treatment['metric']
        )

        # CUPED analysis (with variance reduction)
        cuped_result = self._t_test(
            control['cuped_metric'],
            treatment['cuped_metric']
        )

        # Calculate variance reduction
        variance_reduction = (
            1 - np.var(experiment_data['cuped_metric']) /
            np.var(experiment_data['metric'])
        )

        return {
            'regular_analysis': regular_result,
            'cuped_analysis': cuped_result,
            'theta': theta,
            'variance_reduction_pct': f"{variance_reduction:.1%}",
            'interpretation': self._interpret_cuped(
                regular_result, cuped_result, variance_reduction
            )
        }

    def _t_test(self, control_data, treatment_data):
        """
        Performs independent samples t-test.

        This is like the z-test but for continuous metrics (not just yes/no).
        Examples: revenue per user, time spent, games played
        """

        # Calculate means
        control_mean = control_data.mean()
        treatment_mean = treatment_data.mean()

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(control_data, treatment_data)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (control_data.var() + treatment_data.var()) / 2
        )
        cohens_d = (treatment_mean - control_mean) / pooled_std

        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'difference': treatment_mean - control_mean,
            'percent_change': (treatment_mean - control_mean) / control_mean,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'is_significant': p_value < 0.05
        }

    def _interpret_cuped(self, regular, cuped, var_reduction):
        """
        Explains what CUPED accomplished in plain English.
        """

        regular_sig = "significant" if regular['is_significant'] else "not significant"
        cuped_sig = "significant" if cuped['is_significant'] else "not significant"

        interpretation = {
            'variance_reduced_by': f"{var_reduction:.1%}",
            'regular_result': regular_sig,
            'cuped_result': cuped_sig
        }

        # Did CUPED help us reach significance?
        if not regular['is_significant'] and cuped['is_significant']:
            interpretation['impact'] = "🎯 CUPED found significance! Regular analysis missed it."
            interpretation['explanation'] = (
                f"By reducing variance by {var_reduction:.0%}, CUPED revealed a real "
                "effect that was hidden in the noise. This means you can ship faster!"
            )
        elif regular['is_significant'] and cuped['is_significant']:
            interpretation['impact'] = "✅ Both methods agree - strong result!"
            interpretation['explanation'] = (
                f"CUPED reduced variance by {var_reduction:.0%}, making the result "
                "even more confident. You could have stopped the test earlier."
            )
        else:
            interpretation['impact'] = "❌ No significant effect found."
            interpretation['explanation'] = (
                f"Even with {var_reduction:.0%} variance reduction, no significant "
                "difference detected. The treatment likely doesn't work."
            )

        return interpretation
    
    ################################# BAYESIAN analysis functions ###########################################
    def bayesian_ab_test(self, control_conversions, control_total,
                        treatment_conversions, treatment_total,
                        prior_alpha=1, prior_beta=1):
        """
        Bayesian A/B test for conversion rates.

        Instead of "significant/not significant," this tells you:
        1. Probability that treatment is better than control
        2. Expected value of the improvement
        3. Credible interval (Bayesian version of confidence interval)

        Parameters:
        -----------
        control_conversions: Number who converted in control
        control_total: Total users in control
        treatment_conversions: Number who converted in treatment
        treatment_total: Total users in treatment
        prior_alpha, prior_beta: Prior beliefs (default = uninformative)

        The "prior" represents what you believe BEFORE seeing data.
        Default (1, 1) = "no idea, could be anything"

        Returns:
        --------
        Bayesian analysis results
        """

        # Update our beliefs based on the data (Bayes' Theorem in action!)
        # Posterior = Prior + Data

        control_alpha_post = prior_alpha + control_conversions
        control_beta_post = prior_beta + (control_total - control_conversions)

        treatment_alpha_post = prior_alpha + treatment_conversions
        treatment_beta_post = prior_beta + (treatment_total - treatment_conversions)

        # Generate samples from the posterior distributions
        # This is like simulating thousands of possible conversion rates
        n_samples = 100000

        control_samples = np.random.beta(
            control_alpha_post,
            control_beta_post,
            n_samples
        )

        treatment_samples = np.random.beta(
            treatment_alpha_post,
            treatment_beta_post,
            n_samples
        )

        # Calculate key metrics

        # 1. Probability that treatment beats control
        prob_treatment_better = np.mean(treatment_samples > control_samples)

        # 2. Expected lift
        lift_samples = treatment_samples - control_samples
        expected_lift = np.mean(lift_samples)

        # 3. Credible interval (95%)
        # "We're 95% confident the true lift is in this range"
        credible_interval = np.percentile(lift_samples, [2.5, 97.5])

        # 4. Expected value of treatment
        expected_treatment_rate = np.mean(treatment_samples)
        expected_control_rate = np.mean(control_samples)

        # 5. Risk calculations
        # "If I choose treatment but it's actually worse, how bad could it be?"
        potential_loss_if_wrong = np.mean(
            np.maximum(0, control_samples - treatment_samples)
        )

        return {
            'probability_b_beats_a': prob_treatment_better,
            'probability_b_beats_a_pct': f"{prob_treatment_better:.1%}",
            'expected_lift': expected_lift,
            'expected_lift_pct': f"{expected_lift:.2%}",
            'credible_interval_95': credible_interval,
            'credible_interval_95_pct': f"[{credible_interval[0]:.2%}, {credible_interval[1]:.2%}]",
            'expected_treatment_rate': expected_treatment_rate,
            'expected_control_rate': expected_control_rate,
            'potential_loss': potential_loss_if_wrong,
            'potential_loss_pct': f"{potential_loss_if_wrong:.2%}",
            'recommendation': self._bayesian_recommendation(
                prob_treatment_better, expected_lift, potential_loss_if_wrong
            ),
            'samples': {
                'control': control_samples,
                'treatment': treatment_samples,
                'lift': lift_samples
            }
        }

    def _bayesian_recommendation(self, prob_better, expected_lift, risk):
        """
        Makes a business decision based on Bayesian results.
        """

        # Decision thresholds (you can customize these!)
        HIGH_CONFIDENCE = 0.95  # 95% sure
        MEDIUM_CONFIDENCE = 0.85  # 85% sure
        ACCEPTABLE_RISK = 0.01  # 1% potential loss

        if prob_better > HIGH_CONFIDENCE and expected_lift > 0:
            return {
                'action': 'SHIP IT! 🚀',
                'confidence': 'VERY HIGH',
                'reason': f'{prob_better:.0%} probability of improvement',
                'explanation': (
                    f"Treatment is very likely better ({prob_better:.0%} confidence) "
                    f"with expected lift of {expected_lift:.1%}. Low risk of being wrong."
                )
            }

        elif prob_better > MEDIUM_CONFIDENCE and risk < ACCEPTABLE_RISK:
            return {
                'action': 'SHIP IT ✅',
                'confidence': 'MEDIUM-HIGH',
                'reason': f'{prob_better:.0%} probability with minimal risk',
                'explanation': (
                    f"Treatment is likely better ({prob_better:.0%} confidence) "
                    f"with acceptable downside risk ({risk:.2%}). Reasonable to ship."
                )
            }

        elif prob_better < 0.50:
            return {
                'action': 'DO NOT SHIP ❌',
                'confidence': 'HIGH',
                'reason': f'Only {prob_better:.0%} chance of being better',
                'explanation': (
                    f"Treatment is more likely to be worse ({1-prob_better:.0%} probability). "
                    "Reject this variant."
                )
            }

        else:
            return {
                'action': 'KEEP TESTING ⏳',
                'confidence': 'LOW',
                'reason': 'Insufficient evidence either way',
                'explanation': (
                    f"Probability of improvement is {prob_better:.0%}, which is inconclusive. "
                    "Collect more data or try a different approach."
                )
            }

    def plot_bayesian_results(self, bayesian_results, save_path=None):
        """
        Visualizes Bayesian posterior distributions.

        This creates a beautiful chart showing the probability distributions.
        """

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Posterior distributions of conversion rates
        ax1 = axes[0]

        control_samples = bayesian_results['samples']['control']
        treatment_samples = bayesian_results['samples']['treatment']

        ax1.hist(control_samples, bins=100, alpha=0.6, label='Control',
                color='#FF6B6B', density=True)
        ax1.hist(treatment_samples, bins=100, alpha=0.6, label='Treatment',
                color='#4ECDC4', density=True)

        ax1.axvline(bayesian_results['expected_control_rate'],
                   color='#FF6B6B', linestyle='--', linewidth=2,
                   label=f'Control Mean: {bayesian_results["expected_control_rate"]:.2%}')
        ax1.axvline(bayesian_results['expected_treatment_rate'],
                   color='#4ECDC4', linestyle='--', linewidth=2,
                   label=f'Treatment Mean: {bayesian_results["expected_treatment_rate"]:.2%}')

        ax1.set_xlabel('Conversion Rate', fontsize=12)
        ax1.set_ylabel('Probability Density', fontsize=12)
        ax1.set_title('Posterior Distributions of Conversion Rates',
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Distribution of lift
        ax2 = axes[1]

        lift_samples = bayesian_results['samples']['lift']

        ax2.hist(lift_samples, bins=100, alpha=0.7, color='#95E1D3', edgecolor='black')

        # Mark the credible interval
        ci = bayesian_results['credible_interval_95']
        ax2.axvline(ci[0], color='red', linestyle='--', linewidth=2,
                   label=f'95% Credible Interval: [{ci[0]:.2%}, {ci[1]:.2%}]')
        ax2.axvline(ci[1], color='red', linestyle='--', linewidth=2)

        # Mark zero (no difference)
        ax2.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.5,
                   label='No Difference')

        # Mark expected lift
        ax2.axvline(bayesian_results['expected_lift'], color='green',
                   linestyle='--', linewidth=3,
                   label=f'Expected Lift: {bayesian_results["expected_lift"]:.2%}')

        # Shade the area where treatment is better (lift > 0)
        ax2.axvspan(0, lift_samples.max(), alpha=0.2, color='green',
                   label=f'P(Treatment Better) = {bayesian_results["probability_b_beats_a"]:.1%}')

        ax2.set_xlabel('Lift (Treatment - Control)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Treatment Effect',
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def print_bayesian_results(self, results):
        """
        Pretty-prints Bayesian analysis.
        """
        print("="*70)
        print("BAYESIAN A/B TEST RESULTS")
        print("="*70)

        print("\n🎲 PROBABILITY:")
        print(f"   P(Treatment Beats Control): {results['probability_b_beats_a_pct']}")

        print("\n📈 EXPECTED IMPACT:")
        print(f"   Expected Lift: {results['expected_lift_pct']}")
        print(f"   95% Credible Interval: {results['credible_interval_95_pct']}")

        print("\n📊 EXPECTED RATES:")
        print(f"   Control: {results['expected_control_rate']:.2%}")
        print(f"   Treatment: {results['expected_treatment_rate']:.2%}")

        print("\n⚠️ RISK ANALYSIS:")
        print(f"   Potential Loss if Wrong: {results['potential_loss_pct']}")

        print("\n💡 RECOMMENDATION:")
        rec = results['recommendation']
        print(f"   Action: {rec['action']}")
        print(f"   Confidence: {rec['confidence']}")
        print(f"   Reason: {rec['reason']}")
        print(f"   Explanation: {rec['explanation']}")

        print("="*70)


    ################################# SEQUENTIAL analysis functions ###########################################
    def sequential_test(self, control_conversions, control_total,
                       treatment_conversions, treatment_total,
                       alpha=0.05, spending_function='obrien_fleming'):
        """
        Sequential testing with early stopping rules.

        This lets you check results multiple times during the experiment
        without inflating your false positive rate.

        Parameters:
        -----------
        spending_function: How to "spend" your alpha across multiple looks
            - 'obrien_fleming': Conservative early, aggressive later
            - 'pocock': Equal spending at each look

        Returns:
        --------
        Decision on whether to stop early or continue
        """

        # Calculate current test statistic
        p1 = control_conversions / control_total
        p2 = treatment_conversions / treatment_total

        p_pool = (control_conversions + treatment_conversions) / (control_total + treatment_total)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/control_total + 1/treatment_total))

        z_stat = (p2 - p1) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Calculate what percentage of planned sample we've seen
        # (In real life, you'd track this based on your planned total)
        information_fraction = 0.5  # Example: we're halfway through

        # Calculate adjusted alpha threshold for this look
        adjusted_alpha = self._calculate_spending_alpha(
            information_fraction,
            alpha,
            spending_function
        )

        # Make decision
        should_stop_for_efficacy = p_value < adjusted_alpha and (p2 > p1)
        should_stop_for_futility = self._check_futility(
            z_stat, information_fraction
        )

        return {
            'current_p_value': p_value,
            'adjusted_alpha': adjusted_alpha,
            'information_fraction': information_fraction,
            'information_fraction_pct': f"{information_fraction:.0%}",
            'z_statistic': z_stat,
            'should_stop_for_efficacy': should_stop_for_efficacy,
            'should_stop_for_futility': should_stop_for_futility,
            'recommendation': self._sequential_recommendation(
                should_stop_for_efficacy,
                should_stop_for_futility,
                information_fraction,
                p_value,
                adjusted_alpha
            )
        }

    def _calculate_spending_alpha(self, information_fraction, alpha, function):
        """
        Calculates adjusted alpha threshold for this interim look.

        O'Brien-Fleming: Very strict early, more lenient later
        Pocock: Same threshold at every look
        """

        if function == 'obrien_fleming':
            # O'Brien-Fleming boundary
            # Early looks require very strong evidence
            z_boundary = 2 * stats.norm.ppf(1 - alpha/4) / np.sqrt(information_fraction)
            adjusted_alpha = 2 * (1 - stats.norm.cdf(z_boundary))

        elif function == 'pocock':
            # Pocock boundary
            # Constant threshold (approximately)
            adjusted_alpha = alpha / 2  # Simplified

        else:
            adjusted_alpha = alpha  # No adjustment

        return adjusted_alpha

    def _check_futility(self, z_stat, information_fraction):
        """
        Checks if experiment is unlikely to succeed even if run to completion.

        This saves time by stopping experiments that won't work.
        """

        # Simple futility rule: If z-stat is negative and we're far enough along,
        # it's unlikely to become positive

        if information_fraction > 0.5 and z_stat < -0.5:
            return True

        return False

    def _sequential_recommendation(self, stop_efficacy, stop_futility,
                                   info_frac, p_val, adj_alpha):
        """
        Makes recommendation based on sequential analysis.
        """

        if stop_efficacy:
            return {
                'action': 'STOP & SHIP! 🚀',
                'reason': f'Early efficacy detected ({info_frac:.0%} through test)',
                'explanation': (
                    f"P-value ({p_val:.6f}) is below adjusted threshold ({adj_alpha:.6f}). "
                    f"Strong evidence of positive effect. Safe to stop early and ship."
                )
            }

        elif stop_futility:
            return {
                'action': 'STOP & KILL ⛔',
                'reason': f'Futility detected ({info_frac:.0%} through test)',
                'explanation': (
                    "Current results suggest the treatment is unlikely to succeed "
                    "even if run to completion. Stop and try something else."
                )
            }
        else:
            return {
                'action': 'KEEP RUNNING ⏳',
                'reason': f'Inconclusive at {info_frac:.0%} progress',
                'explanation': (
                    f"Current p-value ({p_val:.6f}) exceeds adjusted threshold ({adj_alpha:.6f}). "
                    "Continue collecting data until planned sample size or next interim analysis."
                )
            }

    def print_sequential_results(self, results):
        """
        Pretty-prints sequential testing results.
        """
        print("="*70)
        print("SEQUENTIAL TESTING ANALYSIS")
        print("="*70)

        print("\n📊 CURRENT STATUS:")
        print(f"   Progress: {results['information_fraction_pct']} of planned sample")
        print(f"   Z-Statistic: {results['z_statistic']:.4f}")
        print(f"   Current P-Value: {results['current_p_value']:.6f}")
        print(f"   Adjusted Alpha Threshold: {results['adjusted_alpha']:.6f}")

        print("\n🚦 STOPPING CRITERIA:")
        print(f"   Stop for Efficacy? {'✅ YES' if results['should_stop_for_efficacy'] else '❌ NO'}")
        print(f"   Stop for Futility? {'✅ YES' if results['should_stop_for_futility'] else '❌ NO'}")

        print("\n💡 RECOMMENDATION:")
        rec = results['recommendation']
        print(f"   Action: {rec['action']}")
        print(f"   Reason: {rec['reason']}")
        print(f"   Explanation: {rec['explanation']}")

        print("="*70)

    def plot_sequential_results(self, sequential_results, save_path=None):
        """
        Creates a visualization of sequential testing results.
        """
        # Create a plot of p-value vs. information fraction
        plt.figure(figsize=(10, 6))
        plt.plot(sequential_results['information_fraction'], sequential_results['current_p_value'], marker='o')
        plt.axhline(y=sequential_results['adjusted_alpha'], color='r', linestyle='--')
        plt.xlabel('Information Fraction')
        plt.ylabel('P-Value')
        plt.title('Sequential Testing Results')
        plt.grid(True)

        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path)

        # Show the plot
        plt.show()

# EXAMPLE USAGE (Default usage)
if __name__ == "__main__":
    # Load the experiment data we created earlier
    BASE_DIR = os.path.dirname(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    experiment_data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'raw', 'pricing_experiment.csv'))

    # Split into control and treatment groups
    control = experiment_data[experiment_data['variant'] == 'control']
    treatment_1 = experiment_data[experiment_data['variant'] == 'treatment_1']  # $11.99

    # Analyze
    analyzer = ABTestAnalyzer(alpha=0.05)
    results = analyzer.analyze_conversion_test(control, treatment_1)

    # Print results
    analyzer.print_results(results)
