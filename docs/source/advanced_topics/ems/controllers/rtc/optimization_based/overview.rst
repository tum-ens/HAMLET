Optimization-based Methods
==========================

General Approach
--------------

The general approach to implementing optimization-based controllers involves:

1. **System modeling**: Defining the mathematical representation of the energy system
2. **Problem formulation**: Specifying the objective function and constraints
3. **Solver selection**: Choosing an appropriate optimization solver
4. **Solution implementation**: Applying the optimal control actions to the system
5. **Receding horizon implementation**: Re-solving the problem at each time step with updated information

Documentation Structure
---------------------

This section is organized as follows:

.. toctree::
   :maxdepth: 2

   mathematical_formulation/overview
   implementation/overview
   build_your_own/overview

The **Mathematical Formulation** section focuses on the general understanding of the objective function and component models, independent of specific implementation details.

The **Implementation** section provides concrete implementations using different frameworks (Linopy and PyOptInterface).

The **Build Your Own** section provides instructions on how to extend or customize the optimization-based controllers for specific needs.