Models
======

HAMLET's simulation framework is built upon detailed mathematical models that define the behavior of key energy system components and decision-making processes. This section provides an in-depth look into the various models used, their assumptions, and mathematical formulations.

HAMLET's models are categorized into two main groups:

1. **Component Models**: These describe the physical behavior of energy system elements, such as energy consumption, generation, and storage.
2. **Home Energy Management System (HEMS)**: This includes forecasting, control strategies, and trading mechanisms that enable decision-making.

Component Models
----------------

HAMLET models various energy system components to simulate decentralized energy systems accurately. The key component models include:

- **Electricity Load**: Models demand patterns based on occupancy, appliance usage, and stochastic variations.
- **PV Generation**: Uses weather data and panel specifications to compute solar power generation.
- **Wind Power**: Simulates wind-based electricity generation based on wind speed distributions and turbine efficiency.
- **Battery Storage**: Captures state-of-charge dynamics, charge/discharge efficiency, and degradation effects.
- **Heat Demand**: Computes thermal energy needs based on building properties, external temperatures, and user preferences.
- **Electric Vehicles**: Models driving patterns, charging behavior, and vehicle-to-grid interactions.
- **Grid Constraints**: Includes voltage and line capacity constraints affecting power flows.

Home Energy Management System (HEMS)
------------------------------------

The HEMS models determine how agents interact with markets, grids, and their own energy assets. It is structured into:

- **Forecasting Methods**: Predict future energy demand, generation, and market prices using:
  - Naïve methods (e.g., persistence models)
  - Statistical models (e.g., SARMA, ARIMA)
  - Machine learning techniques (e.g., Random Forest, Neural Networks)

- **Control Strategies**: Optimize energy usage based on:
  - Rule-Based Control (predefined heuristics)
  - Model Predictive Control (MPC)
  - Reinforcement Learning (self-learning agent behavior)

- **Trading Strategies**: Determine how agents buy and sell energy in markets, including:
  - Zero Intelligence (randomized bids)
  - Linear Bidding (progressively adjusted offers)
  - Retailer-based pricing (buying/selling at fixed prices)

