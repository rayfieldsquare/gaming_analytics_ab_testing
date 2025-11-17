# File: src/simulation.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class ExperimentSimulator:
    """
    Generates realistic user behavior data for A/B tests.

    Why we need this: To practice analysis without real Netflix data.
    """

    def __init__(self, random_seed=42):
        """
        random_seed: Makes results reproducible (same every time)
        """
        np.random.seed(random_seed)

    def simulate_pricing_experiment(self,
                                    n_users_per_variant=10000,
                                    control_price=9.99,
                                    treatment_prices=[11.99, 14.99]):
        """
        Simulates a pricing A/B test.

        Parameters:
        -----------
        n_users_per_variant: How many users see each price (default 10,000)
        control_price: The current price ($9.99)
        treatment_prices: New prices to test ([$11.99, $14.99])

        Returns:
        --------
        DataFrame with columns: user_id, variant, price, converted (yes/no)
        """

        # Key insight: Higher prices = lower conversion but more revenue per user
        # This is called "price elasticity"

        def price_to_conversion_rate(price):
            """
            Models how price affects willingness to pay.
            Lower price = higher conversion.

            This is a realistic model based on e-commerce research.
            """
            base_rate = 0.15  # 15% convert at $9.99
            price_sensitivity = -0.01  # 1% drop per $1 increase
            conversion_rate = base_rate + price_sensitivity * (price - control_price)
            return max(0.05, min(0.25, conversion_rate))  # Keep between 5-25%

        all_data = []

        # Generate control group (original price)
        print("Generating control group...")
        control_conversion_rate = price_to_conversion_rate(control_price)
        control_users = self._generate_users(
            n_users=n_users_per_variant,
            variant_name='control',
            price=control_price,
            conversion_rate=control_conversion_rate
        )
        all_data.append(control_users)

        # Generate treatment groups (new prices)
        print("Generating treatment groups...")
        for i, price in enumerate(treatment_prices):
            treatment_conversion_rate = price_to_conversion_rate(price)
            treatment_users = self._generate_users(
                n_users=n_users_per_variant,
                variant_name=f'treatment_{i+1}',
                price=price,
                conversion_rate=treatment_conversion_rate
            )
            all_data.append(treatment_users)

        # Combine all groups into one dataset
        print("Combining all groups into one dataset...")
        experiment_data = pd.concat(all_data, ignore_index=True)

        # Add realistic metadata
        print("Adding metadata...")
        experiment_data['experiment_id'] = 'pricing_test_001'
        experiment_data['experiment_start_date'] = datetime(2025, 1, 1)
        experiment_data['signup_date'] = [
            experiment_data['experiment_start_date'].iloc[0] + timedelta(days=np.random.randint(0, 30))
            for _ in range(len(experiment_data))
        ]

        return experiment_data

    def _generate_users(self, n_users, variant_name, price, conversion_rate):
        """
        Helper function: Creates individual user records.
        """
        user_ids = [f'user_{variant_name}_{i}' for i in range(n_users)]

        # Randomly determine who converts based on conversion_rate
        # np.random.binomial is like flipping a weighted coin n_users times
        converted = np.random.binomial(1, conversion_rate, n_users)

        df = pd.DataFrame({
            'user_id': user_ids,
            'variant': variant_name,
            'price': price,
            'converted': converted  # 1 = yes, 0 = no
        })

        return df

# EXAMPLE USAGE
if __name__ == "__main__":
    # Create simulator
    simulator = ExperimentSimulator()

    # Run pricing experiment
    experiment_data = simulator.simulate_pricing_experiment(
        n_users_per_variant=10000,
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
    # find the absolute path to the project root
    #  BASE_DIR = os.path.dirname(os.path.dirname(os.path.join(os.path.abspath(__file__))))
    BASE_DIR = os.path.dirname(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    experiment_data.to_csv(os.path.join(BASE_DIR, 'data', 'raw', 'pricing_experiment.csv'), index=False)
    print("\nData saved to data/raw/pricing_experiment.csv")
