import os
from src.analysis import ABTestAnalyzer

# EXAMPLE USAGE
if __name__ == "__main__":
    # Scenario: Testing a new payment flow
    # Control: 1,500 conversions out of 10,000 users (15%)
    # Treatment: 1,750 conversions out of 10,000 users (17.5%)

    analyzer = ABTestAnalyzer()

    bayesian_results = analyzer.bayesian_ab_test(
        control_conversions=1500,
        control_total=10000,
        treatment_conversions=1750,
        treatment_total=10000
    )

    # Print results
    analyzer.print_bayesian_results(bayesian_results)

    # Create visualization
    BASE_DIR = os.path.dirname(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    analyzer.plot_bayesian_results(
        bayesian_results,
        save_path=os.path.join(BASE_DIR, 'data', 'processed', 'bayesian_analysis.png')
    )
