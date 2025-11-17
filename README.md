# 🎮 Gaming Subscription A/B Testing Framework

A comprehensive Python framework for designing, running, and analyzing A/B tests for gaming subscription services. Built to demonstrate production-grade experimentation capabilities for streaming/subscription businesses.

## Project Overview

This project simulates the complete A/B testing lifecycle for a gaming subscription service (similar to Xbox Game Pass or PlayStation Plus), with direct applicability to anyone trying to learn these strategies.

### Key Features

- **Experiment Design Tools**: Sample size calculators, power analysis, MDE estimation
- **Statistical Analysis**: t-tests, z-tests, chi-square, Bayesian methods
- **Advanced Techniques**: CUPED variance reduction, sequential testing, multi-armed bandits
- **Interactive Dashboards**: Plotly visualizations for stakeholder communication
- **Production-Ready Code**: Type hints, docstrings, unit tests

## Quick Start

```bash
# Clone repository
git clone https://github.com/rayfieldsquare/gaming_analytics_ab_testing.git
cd gaming_analytics_ab_testing

# Install dependencies
pip install -r requirements.txt

# Run example experiment
python src/run_experiment.py
```

## Glossary

| Word/Phrase | What It Actually Means | Why It Matters |
| --- | --- | --- |
| **A/B Test** | Showing two different versions of something to people and seeing which works better | The main thing this project is about |
| **Control Group** | The people who see the "old" or "normal" version | You need a baseline to compare against |
| **Treatment Group** | The people who see the "new" version you're testing | These folks get the change you're testing |
| **Conversion Rate** | What percentage of people do the thing you want (like sign up) | The main success metric |
| **Statistical Significance** | Math that tells you if your results are real or just luck | Prevents you from making bad decisions based on randomness |
| **P-Value** | A number between 0 and 1 that measures if results are real (want < 0.05) | Lower = more confident the difference is real |
| **Sample Size** | How many people you need in your test | Too few = unreliable results |
| **Power** | Probability you'll detect a real difference if it exists (aim for 80%) | Prevents missing real improvements |
| **MDE** | Minimum Detectable Effect - smallest change you care about (like "2% improvement") | Helps calculate how many people you need |

