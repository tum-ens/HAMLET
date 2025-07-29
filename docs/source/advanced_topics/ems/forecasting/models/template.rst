This file serves as template for when either adding a complete new class of forecasting methods or a new method

[Method Name] Forecasting Methods
===============================

Introduction
-----------

[Brief description of the forecasting method category, its general approach,
and where it fits in the spectrum of forecasting complexity]

[Subsection for each specific method within this category]

[Method Name (as in agents.yaml)]
-------------

[Description of the method]

Available for
~~~~~~~~~~~~~

[List all components where method is available]

[Component 1] | [Component 2] | ...

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   [Mathematical formula]

where:

   - [Parameter explanation]
   - [Parameter explanation]
   - ...

Configuration
~~~~~~~~~~~~~

[Method name] forecasting methods can be configured in the agent configuration file:

.. code-block:: yaml

   [component-name]:
     fcast:
       method: [method_name]  # Options: [list of available methods in this category]
       [method_name]:
         [parameter1]: [value]  # [explanation]
         [parameter2]: [value]  # [explanation]
         ...

Notes
~~~~~

[Add anything relevant that is not standard, e.g. explanation of how these methods are implemented in HAMLET,
including any specific optimizations or considerations]
