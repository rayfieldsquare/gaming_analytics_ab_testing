import os
from src.analysis import ABTestAnalyzer

# EXAMPLE USAGE
if __name__ == "__main__":
    analyzer = ABTestAnalyzer()

    # Scenario: Checking results halfway through planned experiment
    # We've seen 5,000 users per group (planned: 10,000 per group)

    sequential_results = analyzer.sequential_test(
        control_conversions=750,   # 15% conversion
        control_total=5000,
        treatment_conversions=900,  # 18% conversion (strong signal!)
        treatment_total=5000
    )

    analyzer.print_sequential_results(sequential_results)

    # Create visualization
    BASE_DIR = os.path.dirname(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    analyzer.plot_sequential_results(
        sequential_results,
        save_path=os.path.join(BASE_DIR, 'data', 'processed', 'sequential_analysis.png')
    )
    