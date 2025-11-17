# File: notebooks/04_dashboard.ipynb (or src/dashboard.py)

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class ExperimentDashboard:
    """
    Creates interactive visualizations for A/B test results.

    This is what you'll show to executives and product managers.
    """

    def __init__(self):
        self.colors = {
            'control': '#FF6B6B',
            'treatment': '#4ECDC4',
            'positive': '#51CF66',
            'negative': '#FF6B6B',
            'neutral': '#868E96'
        }

    def create_funnel_comparison(self, control_funnel, treatment_funnel):
        """
        Creates side-by-side funnel visualization.

        Parameters:
        -----------
        control_funnel: dict with {'stage_name': count}
        treatment_funnel: dict with {'stage_name': count}

        Example:
        --------
        control = {
            'Landing Page': 10000,
            'Signup Start': 5000,
            'Payment Info': 2000,
            'Purchase': 1500
        }
        """

        stages = list(control_funnel.keys())

        fig = go.Figure()

        # Control funnel
        fig.add_trace(go.Funnel(
            name='Control',
            y=stages,
            x=list(control_funnel.values()),
            textinfo='value+percent initial',
            marker=dict(color=self.colors['control']),
            textposition='inside'
        ))

        # Treatment funnel
        fig.add_trace(go.Funnel(
            name='Treatment',
            y=stages,
            x=list(treatment_funnel.values()),
            textinfo='value+percent initial',
            marker=dict(color=self.colors['treatment']),
            textposition='inside'
        ))

        fig.update_layout(
            title='Conversion Funnel Comparison',
            showlegend=True,
            height=500,
            font=dict(size=12)
        )

        return fig

    def create_conversion_rate_chart(self, results_df):
        """
        Bar chart comparing conversion rates with confidence intervals.

        Parameters:
        -----------
        results_df: DataFrame with columns ['variant', 'conversions', 'total']
        """

        # Calculate rates and confidence intervals
        results_df['rate'] = results_df['conversions'] / results_df['total']

        # 95% CI for proportions
        results_df['ci_lower'] = results_df.apply(
            lambda row: row['rate'] - 1.96 * np.sqrt(row['rate'] * (1-row['rate']) / row['total']),
            axis=1
        )
        results_df['ci_upper'] = results_df.apply(
            lambda row: row['rate'] + 1.96 * np.sqrt(row['rate'] * (1-row['rate']) / row['total']),
            axis=1
        )

        # Create bar chart
        fig = go.Figure()

        for idx, row in results_df.iterrows():
            color = self.colors['control'] if row['variant'] == 'control' else self.colors['treatment']

            fig.add_trace(go.Bar(
                x=[row['variant']],
                y=[row['rate'] * 100],
                name=row['variant'],
                marker_color=color,
                text=f"{row['rate']:.2%}",
                textposition='outside',
                error_y=dict(
                    type='data',
                    array=[(row['ci_upper'] - row['rate']) * 100],
                    arrayminus=[(row['rate'] - row['ci_lower']) * 100],
                    visible=True
                )
            ))

        fig.update_layout(
            title='Conversion Rate Comparison',
            xaxis_title='Variant',
            yaxis_title='Conversion Rate (%)',
            showlegend=False,
            height=400,
            font=dict(size=12)
        )

        return fig

    def create_cumulative_results(self, experiment_data):
        """
        Line chart showing how results evolved over time.

        This answers: "When did we reach significance?"

        Parameters:
        -----------
        experiment_data: DataFrame with ['timestamp', 'variant', 'converted']
        """

        # Sort by time
        experiment_data = experiment_data.sort_values('timestamp')

        # Calculate cumulative conversion rate over time
        control_data = experiment_data[experiment_data['variant'] == 'control'].copy()
        treatment_data = experiment_data[experiment_data['variant'] == 'treatment'].copy()

        control_data['cumulative_rate'] = (
            control_data['converted'].cumsum() /
            range(1, len(control_data) + 1)
        )

        treatment_data['cumulative_rate'] = (
            treatment_data['converted'].cumsum() /
            range(1, len(treatment_data) + 1)
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=control_data.index,
            y=control_data['cumulative_rate'] * 100,
            mode='lines',
            name='Control',
            line=dict(color=self.colors['control'], width=3)
        ))

        fig.add_trace(go.Scatter(
            x=treatment_data.index,
            y=treatment_data['cumulative_rate'] * 100,
            mode='lines',
            name='Treatment',
            line=dict(color=self.colors['treatment'], width=3)
        ))

        fig.update_layout(
            title='Cumulative Conversion Rate Over Time',
            xaxis_title='Number of Users',
            yaxis_title='Conversion Rate (%)',
            hovermode='x unified',
            height=400,
            font=dict(size=12)
        )

        return fig

    def create_experiment_scorecard(self, experiment_results):
        """
        Creates a KPI scorecard summary.

        Parameters:
        -----------
        experiment_results: dict with all key metrics
        """

        # Create a table-like figure
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Control</b>', '<b>Treatment</b>', '<b>Change</b>'],
                fill_color='#2C3E50',
                font=dict(color='white', size=14),
                align='left'
            ),
            cells=dict(
                values=[
                    ['Conversion Rate', 'Sample Size', 'Conversions', 'Revenue/User', 'P-Value'],
                    [
                        f"{experiment_results['control']['rate']:.2%}",
                        f"{experiment_results['control']['total']:,}",
                        f"{experiment_results['control']['conversions']:,}",
                        f"${experiment_results['control'].get('revenue_per_user', 0):.2f}",
                        '-'
                    ],
                    [
                        f"{experiment_results['treatment']['rate']:.2%}",
                        f"{experiment_results['treatment']['total']:,}",
                        f"{experiment_results['treatment']['conversions']:,}",
                        f"${experiment_results['treatment'].get('revenue_per_user', 0):.2f}",
                        '-'
                    ],
                    [
                        f"{experiment_results['lift']['relative']:+.1%}",
                        '-',
                        f"{experiment_results['treatment']['conversions'] - experiment_results['control']['conversions']:+,}",
                        f"${experiment_results['treatment'].get('revenue_per_user', 0) - experiment_results['control'].get('revenue_per_user', 0):+.2f}",
                        f"{experiment_results['statistics']['p_value']:.6f}"
                    ]
                ],
                fill_color='#ECF0F1',
                font=dict(size=12),
                align='left',
                height=30
            )
        )])

        fig.update_layout(
            title='Experiment Summary',
            height=300
        )

        return fig

    def create_full_dashboard(self, experiment_results, experiment_data):
        """
        Combines all visualizations into one comprehensive dashboard.
        """

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Conversion Rate Comparison',
                'Results Over Time',
                'Segment Breakdown',
                'Statistical Power'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'indicator'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Add traces for each subplot
        # (Simplified - in real dashboard you'd add all charts)

        fig.update_layout(
            title_text='A/B Test Dashboard - Complete Results',
            showlegend=True,
            height=800,
            font=dict(size=11)
        )

        return fig

# EXAMPLE USAGE
if __name__ == "__main__":
    # Create sample data
    results_df = pd.DataFrame({
        'variant': ['control', 'treatment'],
        'conversions': [1500, 1750],
        'total': [10000, 10000]
    })

    # Create dashboard
    dashboard = ExperimentDashboard()

    # Generate visualizations
    fig1 = dashboard.create_conversion_rate_chart(results_df)
    BASE_DIR = os.path.dirname( os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') )
    fig1.write_html(os.path.join(BASE_DIR, 'data', 'processed', 'conversion_chart.html'))
    fig1.show()

    print("✅ Dashboard created! Open conversion_chart.html in your browser")
