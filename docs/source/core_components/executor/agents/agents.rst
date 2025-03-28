Agent Overview
==============

The **Agent** represents an energy system participant, such as a household, commercial entity, or industrial consumer, that interacts with markets and the grid. Each agent follows a structured workflow to make decisions, optimize its energy usage, and engage in trading.

What Does an Agent Do?
----------------------
Each agent in HAMLET:
- **Retrieves grid data** to understand network constraints.
- **Obtains forecasts** to predict future energy consumption, production, and prices.
- **Executes control strategies** via the Energy Management System (EMS).
- **Trades energy** by submitting bids and offers to the market.

Agent Execution Workflow
------------------------
Each simulation step follows a structured sequence:

1. **Grid Data Retrieval**
   - The agent gathers grid-related data to assess constraints and available capacity.

2. **Forecasting**
   - Agents predict their future energy needs and availability using forecasting models.
   - Forecasts can be based on historical data, weather predictions, or machine learning techniques.

3. **Control Strategy Execution**
   - The Energy Management System (EMS) defines how the agent manages its energy usage.
   - The EMS can follow:
     - **Rule-based strategies**
     - **Optimization models (e.g., linear programming)**
     - **Reinforcement learning-based decisions**

4. **Market Participation**
   - Based on its forecast and EMS, the agent submits **bids and offers** to the market.
   - Market clearing determines how much energy is bought or sold.

Agent Structure
---------------
Each agent consists of the following components:

1. **Agent Type**
   - The category of the agent (e.g., **single-family home, multi-family home, industry**).
   - Defines the agent's properties such as load profiles, generation capacity, and flexibility.

2. **Energy Management System (EMS)**
   - The EMS defines how the agent interacts with energy markets and storage systems.
   - Determines when to store, consume, or trade energy.

3. **Trading Strategy**
   - Defines how the agent participates in energy trading.
   - Strategies include:
     - **Retailer-based trading:** Agents buy and sell at retailer prices.
     - **Market-driven strategies:** Agents bid dynamically based on forecasts.
     - **Zero Intelligence (ZI) models:** Randomized trading behaviors.

4. **Grid Interaction**
   - Ensures that the agent's transactions respect grid constraints.
   - If grid limitations exist, the agent may adjust its trading behavior.

Extending Agent Behavior
------------------------
HAMLET allows customization of agent behavior:

- **Custom Forecasting Models**
  - Users can integrate different forecasting techniques, from simple averages to deep learning models.

- **Advanced EMS Control**
  - The EMS can be customized to include complex decision-making mechanisms.

- **New Trading Strategies**
  - Users can define new trading mechanisms beyond the default strategies.

By modeling agents with **autonomous decision-making capabilities**, HAMLET provides a powerful simulation environment for decentralized energy markets.
