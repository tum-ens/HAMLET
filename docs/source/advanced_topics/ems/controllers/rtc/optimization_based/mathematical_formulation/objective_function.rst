Objective Function
==================

This page provides concise explanations of how objective functions work specifically in the Real-Time Control (RTC) context.

In the RTC context, the objective function typically focuses on:

1. **Operational cost minimization**: Minimizing immediate electricity costs based on current prices and forecasted prices for a short horizon.

   .. math::

      f_{\text{cost}}(\mathbf{x}) = \sum_{t=1}^{T} c_{\text{grid}}(t) \cdot P_{\text{grid,import}}(t) - c_{\text{feed-in}}(t) \cdot P_{\text{grid,export}}(t)

   where :math:`T` is typically a short horizon (e.g., 1-24 hours).

2. **Comfort satisfaction**: Maintaining user comfort within acceptable ranges for the immediate future.

   .. math::

      f_{\text{comfort}}(\mathbf{x}) = \sum_{t=1}^{T} w_{\text{comfort}} \cdot (T_{\text{indoor}}(t) - T_{\text{setpoint}}(t))^2

3. **Renewable energy utilization**: Maximizing self-consumption of available renewable generation.

   .. math::

      f_{\text{renewable}}(\mathbf{x}) = \sum_{t=1}^{T} w_{\text{renewable}} \cdot P_{\text{curtailed}}(t)

The RTC objective function is characterized by:

- **Short time horizon**: Focus on immediate or near-term optimization (typically hours rather than days)
- **Computational efficiency**: Formulated for fast solving to enable real-time decision-making
- **Reactive approach**: Emphasis on responding to current conditions rather than long-term planning
- **Simplified formulation**: Less complex than forecast-based controllers to enable faster solution times