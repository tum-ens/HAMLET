Installation
============

HAMLET is a Python-based tool designed for easy setup. Its dependencies are defined once, in
``pyproject.toml``, and pinned to exact versions in the committed ``uv.lock``.

No solver licence is needed to get started: HAMLET installs the open-source solver HiGHS, and
``examples/create_simple_scenario`` is configured to use it. Gurobi is optional.

1. **Prerequisites**:

   - `uv <https://docs.astral.sh/uv/getting-started/installation/>`_
   - An IDE, if you want one (e.g., PyCharm)

   You do not need to install Python yourself; ``uv`` fetches the version HAMLET requires.

2. **Clone the Repository**:
   Clone HAMLET to a local directory using Git:

.. code-block:: bash

   git clone https://github.com/tum-ens/HAMLET.git

3. **Install**:
   From the repository root:

.. code-block:: bash

   uv sync

This creates ``.venv/``, installs the exact versions recorded in ``uv.lock``, and installs HAMLET
itself in editable mode, so ``import hamlet`` works without any ``PYTHONPATH`` setting. Use
``uv run <command>`` to run against that environment without activating it.

4. **Configure IDE** (optional):
   In PyCharm, open HAMLET's directory, then go to
   ``File -> Settings -> Project -> Python Interpreter -> Add -> Existing environment`` and select
   the interpreter in ``.venv/``.

5. **Optional components**:
   Extras are installed only on request:

.. code-block:: bash

   uv sync --extra tensorflow   # the two neural-network forecast models
   uv sync --extra gurobi       # the Gurobi backend
   uv sync --extra notebooks    # Jupyter, for the example notebooks

6. **Install Gurobi** (optional):

   - If you do not have a license yet, you might be able to acquire one through your university by visiting `https://www.gurobi.com` and create an account using your university email.
   - Download the latest Gurobi version.
   - Follow the Academic License instructions for activation.
   - Set ``solver: gurobi`` under ``optimization`` in the scenario's ``agents.yaml``.

7. **Test Your Installation**:
   Run the simple example scenario, which needs no solver licence:

.. code-block:: bash

   uv run --extra notebooks jupyter notebook examples/create_simple_scenario/run.ipynb
