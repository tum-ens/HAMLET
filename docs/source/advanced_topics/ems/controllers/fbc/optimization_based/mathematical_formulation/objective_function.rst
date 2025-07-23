Objective Function
================


In forecast-based controllers, the objective function typically spans multiple timesteps in the prediction horizon. The general form is:

.. math::

   \min_{\mathbf{x}} \sum_{t=0}^{T-1} f_t(\mathbf{x}_t)

where:
- :math:`\mathbf{x}_t` is the vector of decision variables at timestep :math:`t`
- :math:`f_t(\mathbf{x}_t)` is the objective function at timestep :math:`t`
- :math:`T` is the optimization horizon

This multi-period structure allows the controller to make decisions that are optimal over the entire horizon, rather than just for the current timestep.

Common Objective Components
-------------------------

The objective function typically includes multiple terms representing different goals, such as:

1. **Cost Minimization**: Minimize the cost of energy consumption and/or maximize the profit from energy sales over the entire horizon.

   .. math::

      f_{\text{cost}}(\mathbf{x}) = \sum_{t=0}^{T-1} \sum_{i} c_{i,t} \cdot x_{i,t}

   where :math:`c_{i,t}` is the cost coefficient for decision variable :math:`x_{i,t}` at timestep :math:`t`.

2. **Comfort Maximization**: Maximize user comfort by minimizing deviations from desired temperature, lighting, or other comfort parameters over time.

   .. math::

      f_{\text{comfort}}(\mathbf{x}) = \sum_{t=0}^{T-1} \sum_{j} w_j \cdot (x_{j,t} - x_{j,t}^{\text{desired}})^2

   where :math:`w_j` is the weight for comfort parameter :math:`j`.

3. **Environmental Impact**: Minimize carbon emissions or other environmental impacts over the horizon.

   .. math::

      f_{\text{env}}(\mathbf{x}) = \sum_{t=0}^{T-1} \sum_{k} e_{k,t} \cdot x_{k,t}

   where :math:`e_{k,t}` is the emission coefficient for decision variable :math:`x_{k,t}` at timestep :math:`t`.

4. **Terminal Cost**: Penalize undesirable terminal states or reward desirable ones.

   .. math::

      f_{\text{terminal}}(\mathbf{x}) = \phi(x_{T-1})

   where :math:`\phi` is a function that evaluates the quality of the terminal state.

Multi-Objective Formulation
-------------------------

The overall objective function is typically a weighted sum of these individual objectives:

.. math::

   f(\mathbf{x}) = w_{\text{cost}} \cdot f_{\text{cost}}(\mathbf{x}) + w_{\text{comfort}} \cdot f_{\text{comfort}}(\mathbf{x}) + w_{\text{env}} \cdot f_{\text{env}}(\mathbf{x}) + w_{\text{terminal}} \cdot f_{\text{terminal}}(\mathbf{x})

where :math:`w_{\text{cost}}`, :math:`w_{\text{comfort}}`, :math:`w_{\text{env}}`, and :math:`w_{\text{terminal}}` are weights that reflect the relative importance of each objective.

The choice of weights depends on the specific application and user preferences. For example, in a residential setting, comfort might be weighted more heavily, while in a commercial setting, cost might be the primary concern.

Forecast Integration
-----------------

A key aspect of objective functions in FBCs is the integration of forecasts. Cost coefficients, comfort requirements, and environmental factors may all vary over the prediction horizon based on forecasts:

- **Price Forecasts**: Future electricity or heat prices affect the cost minimization term
- **Weather Forecasts**: Future outdoor temperatures affect building thermal dynamics and comfort
- **Load Forecasts**: Predicted energy demands affect system operation and constraints
- **Renewable Generation Forecasts**: Predicted availability of renewable energy affects optimal scheduling

The quality of these forecasts directly impacts the performance of the FBC. Better forecasts generally lead to better decisions, but the controller should also be robust to forecast errors.

Implementation Considerations
---------------------------

When implementing objective functions for FBCs, several practical considerations should be taken into account:

1. **Normalization**: Different objective components may have different units and scales. Normalization helps ensure that the weighted sum is meaningful.

2. **Linearization**: For MILP solvers, nonlinear terms (e.g., quadratic comfort penalties) may need to be linearized.

3. **Time-Varying Weights**: In some applications, the relative importance of different objectives may vary over the prediction horizon.

4. **Discount Factors**: Future costs or benefits may be discounted to prioritize near-term performance.

5. **Robustness**: The objective function may include terms to handle uncertainty in forecasts or system parameters.

Customizing Objective Functions
-----------------------------

Users can customize the objective function by:

1. Adjusting the weights of existing objectives
2. Adding new objective terms
3. Implementing custom objective functions

See the "Build Your Own" section for detailed instructions on how to customize objective functions for specific applications.