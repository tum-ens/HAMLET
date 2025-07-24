Installation
============

HAMLET is a Python-based tool designed for easy setup. This guide explains how to install HAMLET using Anaconda, PyCharm, and Gurobi. Other IDEs and package managers may also be used.

1. **Prerequisites**:

   - IDE: (e.g., PyCharm)
   - Package Manager: (e.g., Anaconda)
   - Solver: Gurobi (recommended) or HiGHS.

2. **Clone the Repository**:
   Clone HAMLET to a local directory using Git:
.. code-block:: bash

   git clone https://github.com/tum-ewk/hamlet.git

3. **Set Up a Virtual Environment**:
   Use Anaconda to create and activate an environment:
.. code-block:: bash

   conda env create -f ./hamlet/env.yml

4. **Configure IDE**:
   In PyCharm:

   - Open HAMLET's directory.
   - Go to `File -> Settings -> Project -> Python Interpreter`.
   - Add the new environment.

5. **Install Gurobi** (optional):

   - If you do not have a license yet, you might be able to acquire one through your university by visiting `https://www.gurobi.com` and create an account using your university email.
   - Download the latest Gurobi version.
   - Follow the Academic License instructions for activation.

6. **Test Your Installation**:
   Run simple example scenario:
.. code-block:: bash

   examples/create_simple_scenario/run.ipynb
