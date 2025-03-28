Market Overview
===============

Markets in HAMLET facilitate the exchange of energy between agents using different market mechanisms. This section provides an overview of the market structure, types, and how they function within the simulation.

The Role of Markets in HAMLET
-----------------------------
Markets serve as the **central mechanism** where agents can buy and sell energy. They allow for various market designs, including **local energy markets (LEMs)**, **retail transactions**, and **balancing mechanisms**. Key functions of markets include:

- **Energy Matching**: Ensuring supply and demand meet through auctions or direct transactions.
- **Price Formation**: Establishing energy prices based on bids, offers, and retailer tariffs.
- **Grid Interaction**: Incorporating grid constraints, fees, and levies into transactions.

Market Components
-----------------
A HAMLET market consists of the following key elements:

- **Market Type**: Defines the kind of energy being traded (e.g., electricity, heat, hydrogen).
- **Clearing Type**: Determines whether transactions occur **before (ex-ante)** or **after (ex-post)** energy delivery.
- **Market Mechanism**: Specifies how prices are set and energy is allocated.
- **Retailer Pricing**: Sets the default price for agents who do not trade on the market.
- **Grid Fees and Levies**: Includes costs associated with using the grid.

Types of Markets in HAMLET
--------------------------
HAMLET supports different types of markets, each catering to specific trading mechanisms:

### 1. **Local Energy Markets (LEMs)**
   - Enable **peer-to-peer (P2P) trading** among agents.
   - Support different market clearing mechanisms:
     - **Periodic Double Auction (PDA)** – Agents submit bids/offers, and trades are cleared at a uniform price.
     - **Community-Based Clearing** – Prices and trades are determined based on local community agreements.
   - Can operate **independently** or be **coupled** to higher-level markets.

### 2. **Retail Transactions**
   - If an agent does not trade in the local market, it **buys or sells energy from a retailer**.
   - Retail prices can be:
     - **Fixed** – Set in advance and remain constant.
     - **Time-Varying** – Defined by external price files.

### 3. **Balancing Markets**
   - Handle **imbalances** when an agent’s energy production/consumption differs from its market commitments.
   - Balancing costs incentivize agents to **adhere to their trades**.

Market Clearing Mechanisms
--------------------------
HAMLET implements **different clearing mechanisms** to match supply and demand:

- **Ex-Ante Clearing**: Markets clear **before** energy delivery, based on bids/offers (e.g., periodic double auctions).
- **Ex-Post Clearing**: Trades are settled **after** energy has been produced or consumed (e.g., community-based clearing).

Grid and Pricing Considerations
-------------------------------
Markets also incorporate **grid fees, levies, and balancing costs**, which affect trading decisions:

- **Grid Fees**: Costs for using the grid, which may vary depending on the market level.
- **Levies**: Taxes or policy-based charges applied to transactions.
- **Balancing Costs**: Additional fees when agents fail to meet their commitments.
