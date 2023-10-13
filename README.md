HAMLET
=======

**H**ierarchical **A**gent-based **M**arkets for **L**ocal **E**nergy **T**rading

An open-source tool for the agent-based development and testing of energy market applications at a local level.

IN DEVELOPMENT: Please see the branch "develop" for most recent version.

## Description

TBD

## Features
HAMLET offers...
* a fully open-source, agent-based local energy market modelling toolbox
* a modular and extendable design for easy adaptation to your own research questions
* real-time capabilities for the development and testing of hard- and software
* a database-agnostic approach that enables the integration of multiple database technologies
* integrated time-series data for several plant types (household loads, pv, wind, heat pumps, electric vehicles etc...)
* template functionality for load and generation forecasting, trading strategies, cutting-edge market clearing
  algorithms designed specifically for LEMs [paper under review] and much more...

so you only need to adapt the components you want to investigate and/or improve on

## Installation
HAMLET is completely based on Python to keep the installation process simple. This installation guide will
explain how to get HAMLET to run using PyCharm, Gurobi and Anaconda as example. However, other IDEs and package managers
are perfectly suitable as well.

#### Install the following software:
	- IDE: e.g. PyCharm
	- Package Manager: e.g. Anaconda
	- Solver: e.g. Gurobi* or CPLEX. GLPK can be used although this is non-ideal.

    *Installation explained later in this README

#### Clone repository
You can download or clone the repository to a local directory of your choice. You can use version control tools such as 
GitHub Desktop, Sourcetree, GitKraken or pure Git. The link for pure Git is: 

`git clone https://github.com/tum-ewk/hamlet.git`

If using PyCharm, clone the repository, for example, to `./PyCharmProjects/hamlet/`
###
#### Create a virtual python environment
	- Open the AnacondaPrompt.
	- Type `conda env create -f ./PycharmProjects/hamlet/env.yml`
	- Take care to set the correct (absolute) path to your cloned repository.

#### Activate the environment
	- Open PyCharm
	- Go to 'File->Open'
	- Navigate to PyCharmProjects and open hamlet
	- When the project has opened, go to 
         `File->Settings->Project->Python Interpreter->Show all->Add->Conda Environment
          ->Existing environment->Select folder->OK`

#### Install a solver (we recommend Gurobi)
	- Go to gurobi.com
	- Create an account with your university email 
	- When the account has been activated, log in and download the newest Gurobi solver.
	- Go to Academia->Academic Program and Licenses
	- Follow the installation instructions under "Individual Academic Licenses" to activate your copy of Gurobi

### Test your installation
    - Navigate to ./PycharmProjects/hamlet/01 - examples
	- Execute 01_create_scenario.py, followed by 02_execute_scenario.py
    - When the simulation has completed (this may take some time, depending on your system), 
      analyze the results by executing 03_analyze_scenario.py
	- Look at the output plots under hamlet/05 - results/example_singlemarket/analysis/

Your installation was successful if you see plots similar to the following:

TBD

## Contact
Feel free to contact us if you want to contribute to this project, cooperate on an interesting research question
or just to ask about the project.

[Markus Doepfert](https://campus.tum.de/tumonline/ee/ui/ca2/app/desktop/#/pl/ui/$ctx/visitenkarte.show_vcard?$ctx=design=ca2;header=max;lang=de&pPersonenGruppe=3&pPersonenId=99801BCF1F13B4C9)  
markus.doepfert@tum.de  
Research Associate @ TUM ENS

## References

TBD

## License

Copyright (C) 2023 TUM-ENS

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
