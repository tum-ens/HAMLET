Implementations
===============

Introduction
-----------

This section documents the current implementations of optimization-based controllers in HAMLET. These implementations translate the mathematical formulations described in the previous section into concrete, executable code that can be used to solve real-time control problems in energy management systems.


Available Implementations
------------------------

HAMLET currently provides two main implementations for optimization-based controllers:

1. **Linopy Implementation**: A high-level implementation using the Linopy package, which provides a convenient interface for formulating and solving linear and mixed-integer linear programming problems. This implementation leverages Linopy's capabilities for handling labeled dimensions and coordinates, making the code more readable and maintainable.

2. **PyOptInterface Implementation**: A more flexible, lower-level implementation that provides direct access to various optimization solvers. This approach offers more customization options and allows for solver-specific optimizations, but requires more detailed knowledge of the underlying optimization problem.

Both implementations solve the same fundamental optimization problem (optimizing for costs) but differ in their approach, flexibility, and ease of use. The choice between them depends on factors such as the complexity of the problem, the need for customization, and familiarity with the respective frameworks. They also serve as examples on how to implement one's own controllers into the framework.

.. toctree::
   :maxdepth: 2

   linopy
   pyoptinterface