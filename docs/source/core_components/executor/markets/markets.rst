Market Overview
===============

The **Market** in HAMLET serves as a central entity where agents trade energy based on predefined rules and mechanisms. Markets facilitate energy exchange while enforcing pricing models, clearing mechanisms, and grid constraints.

What Does a Market Do?
----------------------
Markets in HAMLET:
- **Collect Bids and Offers** from participating agents.
- **Clear the Market** using predefined clearing methods.
- **Settle Transactions** based on prior commitments and actual energy delivery.
- **Handle Retailer Interactions** for residual energy that is not cleared in the market.
- **Coordinate Across Multiple Market Levels** via market coupling mechanisms.

Market Execution Workflow
-------------------------
A market follows a structured sequence at each simulation timestep:

1. **Retrieve Bids and Offers**
   - Agents submit their bids (buy orders) and offers (sell orders) to the market.
   - Retailer prices are retrieved to act as a fallback in case market transactions fail.

2. **Apply Market Actions**
   - **Market Clearing (`clear`)**: Determines which bids and offers match based on pricing mechanisms.
   - **Market Settlement (`settle`)**: Finalizes transactions after energy is delivered.

3. **Determine Pricing Mechanisms**
   - **Uniform Pricing**: All participants pay/receive the same clearing price.
   - **Discriminatory Pricing**: Participants pay/receive prices based on their individual bids/offers.

4. **Market Coupling (Optional)**
   - If the market is coupled with another market (e.g., local-to-wholesale), uncleared positions are transferred.

Market Structure
----------------
Each market is defined by the following components:

1. **Market Type**
   - Defines what kind of energy is traded (e.g., **electricity, heat, hydrogen**).

2. **Clearing Type**
   - Determines when market clearing occurs:
     - **Ex-Ante**: Market clearing happens before energy delivery.
     - **Ex-Post**: Market clearing happens after energy delivery.

3. **Clearing Method**
   - Specifies how bids and offers are matched:
     - **Periodic Double Auction (PDA)**: Participants submit bids and offers, and a market clearing price is determined.
     - **Community-Based Clearing**: Energy is shared within a defined community before external trading.
     - **No Local Clearing**: All energy is traded with a retailer at set prices.

4. **Pricing Method**
   - Defines how prices are set:
     - **Uniform Pricing**: A single price is set for all transactions.
     - **Discriminatory Pricing**: Participants pay/receive based on their bids.

5. **Market Coupling**
   - Defines how a market interacts with other market levels:
     - **No Coupling**: Unmatched bids/offers are not forwarded.
     - **Above Market Coupling**: Unmatched bids/offers are forwarded to a higher-level market.
     - **Below Market Coupling**: Unmatched bids/offers are passed to a lower-level market.

Extending Market Mechanisms
---------------------------
HAMLET allows customization of market operations:

- **New Market Clearing Rules**
  - Users can define alternative market clearing mechanisms.

- **Dynamic Pricing Strategies**
  - Custom pricing rules can be implemented beyond the default uniform/discriminatory pricing models.

- **Flexible Market Coupling**
  - Developers can integrate additional market levels and configure how markets interact.

By providing a modular structure, HAMLET enables researchers and energy professionals to simulate **diverse market structures** and test the impact of **various pricing and clearing mechanisms** in decentralized energy trading.
