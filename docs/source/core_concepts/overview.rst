Overview
===============

HAMLET (Hierarchical Agent-based Markets for Local Energy Trading) is an advanced simulation framework designed to model decentralized energy systems using agent-based modeling principles. This section provides a conceptual foundation for how HAMLET works and how it differs from traditional energy modeling approaches.

The Need for Agent-Based Energy System Modeling
-----------------------------------------------
Traditional energy system models often rely on centralized optimization approaches that assume perfect information and rational decision-making by all participants. While effective for large-scale planning, these models struggle to capture the dynamics of decentralized and flexible energy markets, such as:

- **Local Energy Markets (LEMs)** where prosumers directly trade energy.
- **Sector Coupling** where different energy vectors (electricity, heat, and hydrogen) interact dynamically.
- **Individual Decision-Making** where consumers and prosumers make decisions based on local conditions and incentives.

Agent-based modeling (ABM) overcomes these limitations by simulating **individual decision-makers** (agents) that operate based on rules, incentives, and learning mechanisms.

HAMLET's Conceptual Approach
----------------------------
HAMLET adopts a structure, based on three core layers:

1. **Agents** - Represent energy system participants (e.g., households, industrial consumers, producers) that make decentralized decisions.
2. **Markets** - Facilitate energy exchanges between agents, applying market mechanisms such as auctions and bilateral trading, or allowing direct purchasing from a retailer (bypassing market clearing).
3. **Grids** - Provide the physical infrastructure where energy flows are simulated, ensuring technical constraints are respected.

This modular structure allows for high flexibility in modeling various market designs and agent behaviors without having to modify the core simulation engine.

How HAMLET Differs from Traditional Models
------------------------------------------
Many classical energy models use **top-down optimization**, assuming a central operator makes decisions for the entire system. HAMLET, in contrast, employs a **bottom-up agent-based approach**, enabling:

- **Heterogeneous agents**: Participants operate based on individual objectives (e.g., minimizing costs, maximizing self-consumption).
- **Distributed decision-making**: Instead of a central planner, multiple agents interact dynamically, reacting to price signals and system conditions.
- **Multiple market types**: HAMLET supports different market designs, such as wholesale or local energy markets, enabling the study of diverse market designs.
- **Scenario-driven simulations**: Users can configure different agent behaviors, market mechanisms, and grid constraints to study real-world energy systems and develop better policies.

By leveraging these capabilities, HAMLET provides a powerful environment for researchers, policymakers, and engineers to explore decentralized energy systems in a realistic and customizable manner.
