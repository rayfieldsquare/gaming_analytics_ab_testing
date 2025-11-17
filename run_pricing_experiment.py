import os
import pandas as pd
from src.simulation import ExperimentSimulator
from src.analysis import ABTestAnalyzer

# Generate data
simulator = ExperimentSimulator()

# Run pricing experiment
experiment_data = simulator.simulate_pricing_experiment(
    n_users_per_variant=12000,
    control_price=9.99,
    treatment_prices=[11.99, 14.99]
)

# Quick summary
print("Experiment Data Summary:")
print(experiment_data.groupby('variant').agg({
    'converted': ['sum', 'mean'],
    'price': 'first'
}))

# Save to CSV
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'pricing_experiment.csv')
experiment_data.to_csv(DATA_FILE, index=False)
print("\nData saved to data/raw/pricing_experiment.csv")

# Load the experiment data we created earlier
experiment_data = pd.read_csv(DATA_FILE)

# Split into control and treatment groups
control = experiment_data[experiment_data['variant'] == 'control']
treatment_1 = experiment_data[experiment_data['variant'] == 'treatment_1']  # $11.99

# Analyze
analyzer = ABTestAnalyzer(alpha=0.05)
results = analyzer.analyze_conversion_test(control, treatment_1)

# Print results
analyzer.print_results(results)