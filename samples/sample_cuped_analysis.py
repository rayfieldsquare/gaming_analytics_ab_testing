import os
import pandas as pd
from src.analysis import ABTestAnalyzer

# EXAMPLE USAGE
if __name__ == "__main__":
    # Simulate experiment data with pre-experiment context
    np.random.seed(42)
    n_users = 5000

    # Generate synthetic data
    data = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(n_users)],
        'variant': ['control'] * (n_users//2) + ['treatment'] * (n_users//2),
        'pre_exp_games_played': np.random.poisson(10, n_users),  # Before experiment
    })

    # Simulate post-experiment metric (correlated with pre-experiment behavior)
    # Treatment adds +1.5 games on average
    data['metric'] = (
        data['pre_exp_games_played'] * 0.8 +  # Past behavior is predictive!
        np.random.normal(0, 3, n_users) +      # Plus some random noise
        (data['variant'] == 'treatment') * 1.5  # Treatment effect
    )

    # Analyze
    analyzer = ABTestAnalyzer()
    results = analyzer.analyze_with_cuped(
        data,
        pre_experiment_metric='pre_exp_games_played'
    )

    # Print results
    print("="*70)
    print("CUPED VARIANCE REDUCTION ANALYSIS")
    print("="*70)
    print(f"\n📉 Variance Reduced By: {results['variance_reduction_pct']}")
    print(f"📊 Theta (adjustment factor): {results['theta']:.4f}")

    print("\n🔴 REGULAR ANALYSIS (No CUPED):")
    reg = results['regular_analysis']
    print(f"   Control Mean: {reg['control_mean']:.2f}")
    print(f"   Treatment Mean: {reg['treatment_mean']:.2f}")
    print(f"   Difference: {reg['difference']:.2f}")
    print(f"   P-Value: {reg['p_value']:.6f}")
    print(f"   Significant? {'✅ YES' if reg['is_significant'] else '❌ NO'}")

    print("\n🟢 CUPED ANALYSIS (Variance Reduced):")
    cup = results['cuped_analysis']
    print(f"   Control Mean: {cup['control_mean']:.2f}")
    print(f"   Treatment Mean: {cup['treatment_mean']:.2f}")
    print(f"   Difference: {cup['difference']:.2f}")
    print(f"   P-Value: {cup['p_value']:.6f}")
    print(f"   Significant? {'✅ YES' if cup['is_significant'] else '❌ NO'}")

    print("\n💡 INTERPRETATION:")
    interp = results['interpretation']
    print(f"   {interp['impact']}")
    print(f"   {interp['explanation']}")
    print("="*70)