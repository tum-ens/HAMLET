Agents in HAMLET
================

In HAMLET, agents represent energy system participants such as households, industrial consumers, and renewable generators. Each agent makes independent decisions about energy consumption, production, and trading based on predefined rules, forecasts, and market conditions.

What Are Agents?
---------------
Agents are the fundamental units of HAMLET's simulation framework. They interact with markets and grids, making decentralized decisions based on:

- **Energy Demand**: Households, industries, and commercial buildings consume electricity and heat.
- **Energy Production**: Renewable energy generators, such as photovoltaic (PV) and wind power, inject electricity into the grid.
- **Storage & Flexibility**: Batteries, heat storage, and electric vehicles (EVs) enable demand-side flexibility.
- **Market Participation**: Agents engage in local energy markets, wholesale markets, or buy directly from retailers.
- **Control & Optimization**: Some agents use advanced controllers (real-time control, forecast-based control) to optimize their behavior.

Agent Categories
----------------
HAMLET supports various agent types, each with unique characteristics and roles:

1. **Single-Family Households (SFH)**: Represent single-family homes with different heating efficiencies and energy consumption profiles.
2. **Commercial, Trading, Services & Public (CTSP)**: Large consumers that may have flexible demand or onsite renewable generation.
3. **Industry**: Specific industries with unique energy consumption patterns and flexibility.
3. **Produvers**: Include photovoltaic (PV) systems and wind turbines that generate electricity based on weather conditions.
4. **Flexibility Providers**: Flexibility providers can adjust their energy consumption or production to respond to market signals.
   - **Batteries** store excess electricity for later use.

Agent Decision-Making
---------------------
Each agent follows a decision-making process influenced by its objectives, market prices, forecasts, and available resources:

- **Baseline Consumption & Production**: Agents determine their energy demand and generation at each timestep.
- **Forecasting**: Agents predict future energy demand, generation, and market prices based on historical data and machine learning models.
- **Energy Trading**: If participating in markets, agents generate bids and offers based on their expected needs and available resources.
- **Grid Interaction**: Agents must respect grid constraints, such as voltage limits and transformer capacities.
