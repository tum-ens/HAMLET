Analyzer Module
===============

The **Analyzer** in HAMLET serves as the primary tool for **evaluating, visualizing, and interpreting** simulation results. It provides insights into agent behaviors, market dynamics, and grid performance, enabling users to extract meaningful conclusions from their simulations.

.. toctree::
   :maxdepth: 1
   :caption: Executor

   agents
   markets
   grids

Purpose of the Analyzer
-----------------------

Unlike the **Creator**, which sets up scenarios, and the **Executor**, which runs simulations, the **Analyzer** focuses on the post-processing phase. It helps answer questions such as:

- How did individual agents behave in the market?
- Were there any grid congestions or stability issues?
- What were the price fluctuations in the market?
- How efficiently was energy traded or utilized?

Key Roles of the Analyzer
-------------------------

1. **Data Processing** – Aggregates and cleans simulation results for easier interpretation.
2. **Visualization** – Generates plots, graphs, and heatmaps to represent trends and patterns.
3. **Performance Evaluation** – Assesses market efficiency, grid performance, and agent profitability.
4. **Exporting Reports** – Allows users to export results in structured formats for further use.

How the Analyzer Fits into the Simulation Workflow
--------------------------------------------------

1. **Scenario Execution** – The **Executor** runs the simulation and stores results.
2. **Result Extraction** – The **Analyzer** fetches simulation data from stored files.
3. **Data Processing & Visualization** – The **Analyzer** applies filters, statistical methods, and plotting tools to make the results interpretable.
4. **User Interpretation** – Users analyze reports, identify patterns, and refine their simulation inputs if necessary.

