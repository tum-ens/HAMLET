Implementations
===============

Introduction
-----------

The implementations handle the additional complexity of FBCs, including multi-period optimization, temporal coupling constraints, and forecast integration. They provide efficient ways to formulate and solve optimization problems over a prediction horizon.

Available Implementations
------------------------

HAMLET currently provides two main implementations for optimization-based forecast-based controllers:

1. **Linopy Implementation**: A high-level implementation using the Linopy package, which provides a convenient interface for formulating and solving linear and mixed-integer linear programming problems over multiple timesteps. This implementation leverages Linopy's capabilities for handling labeled dimensions and coordinates, making it particularly well-suited for the multi-dimensional nature of FBC problems.

2. **PyOptInterface Implementation**: A more flexible, lower-level implementation that provides direct access to various optimization solvers for multi-period problems. This approach offers more customization options and allows for solver-specific optimizations, but requires more detailed knowledge of the underlying optimization problem structure.

Both implementations solve the same fundamental optimization problem (optimizing over a prediction horizon) but differ in their approach, flexibility, and ease of use. The choice between them depends on factors such as the complexity of the problem, the need for customization, and familiarity with the respective frameworks. They also serve as examples on how to implement one's own forecast-based controllers into the framework.

The following pages provide detailed documentation of each implementation approach:

.. toctree::
   :maxdepth: 2

   linopy
   pyoptinterface