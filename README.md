HAMLET
=======

**H**ierarchical **A**gent-based **M**arkets for **L**ocal **E**nergy **T**rading

An open-source tool for the agent-based development and testing of energy market applications (at a local level).

## Description

Due to the increasing complexity of energy systems, the need for more detailed and realistic models has arisen. 
Agent-based models are a promising approach to model the behavior of individual actors in energy systems. For this 
purpose, we developed HAMLET, a modular and extendable open-source toolbox for the development and testing of market 
designs with a focus on local interactions at the distribution level. HAMLET is designed to be used by researchers 
to investigate the interactions between market participants and the impact of different market designs on the technical 
and economic performance of energy systems.

HAMLET was developed with extendability and modularity in mind. Therefore, each functionality can be easily swapped for 
another to test, for example, other market clearing algorithms, trading strategies, or control strategies. The aim of 
the tool is to provide a common platform for researchers to develop and test their own market designs and to compare 
their results with other market designs.

The documentation is available at [hamlet-ens.readthedocs.io](https://hamlet-ens.readthedocs.io); its
sources live in `docs/`.

HAMLET was developed and is maintained by the 
[Chair of Renewable and Sustainable Energy Systems](https://www.epe.ed.tum.de/en/ens/homepage/) of the [Technical
University of Munich](https://www.tum.de/en/). Version 0.1 (February 2024) was developed as part of the research project 
[STROM](https://www.epe.ed.tum.de/en/ens/research/projects/current-projects/strom-sp-3/), funded by the Bavarian Ministry of Economic Affairs, 
Regional Development and Energy.

## Features
HAMLET offers...
* a fully open-source, agent-based energy market modelling toolbox
* a modular and extendable design for easy adaptation to your own research questions
* integrated time-series data for several plant types (household loads, pv, wind, heat pumps, electric vehicles etc...)
* template functionality for load and generation forecasting, trading strategies, market clearing
  algorithms and control strategies and much more...

so you only need to adapt the components you want to investigate and/or improve on

## Installation
HAMLET is completely based on Python to keep the installation process simple. Its dependencies are
defined once, in `pyproject.toml`, and pinned to exact versions in the committed `uv.lock`.

No solver installation is required to get started: HAMLET installs the open-source solver HiGHS as
part of its environment, and `examples/create_simple_scenario` — the example used to test your
installation below — is configured to use it. A commercial solver such as Gurobi* or CPLEX is
optional and is usually faster on larger scenarios; the remaining examples are configured for
Gurobi.

    *Installation explained later in this README

#### Clone repository
You can download or clone the repository to a local directory of your choice. You can use version control tools such as 
GitHub Desktop, Sourcetree, GitKraken or pure Git. The link for pure Git is: 

`git clone https://github.com/tum-ens/HAMLET.git`

If you only want to run HAMLET rather than work on its history, add `--depth=1`: a full clone
fetches about 350 MB of history, a shallow one about 13 MB. Either way the working tree is the
same, so the checkout is ~90 MB rather than ~430 MB.

`git clone --depth=1 https://github.com/tum-ens/HAMLET.git`

Development happens on GitLab at [gitlab.lrz.de/tum-ens/HAMLET](https://gitlab.lrz.de/tum-ens/HAMLET);
the GitHub repository above is a mirror of it.

#### Create the environment

HAMLET uses [uv](https://docs.astral.sh/uv/) ([installation
instructions](https://docs.astral.sh/uv/getting-started/installation/)). From the repository root:

```bash
uv sync
```

That is the whole installation. It creates `.venv/`, fetches Python 3.11 if you do not already
have it, installs the exact versions recorded in `uv.lock`, and installs HAMLET itself in editable
mode — so `import hamlet` works from any directory, with no `PYTHONPATH` and no `sys.path` lines
in your scripts.

Run things through `uv run`, which uses that environment without your having to activate it:

```bash
uv run python -m pytest        # the fast test tier
uv run python run.py
```

Optional components are extras, installed only if you ask for them:

```bash
uv sync --extra tensorflow     # the two neural-network forecast models (~600 MB)
uv sync --extra gurobi         # the Gurobi backend; see the next section
uv sync --extra notebooks      # Jupyter, for examples/*/run.ipynb
```

<details>
<summary>Prefer conda, or plain pip?</summary>

Both work. Neither reads `uv.lock`, so they resolve whatever is current rather than what was
tested — use `uv sync` if you want the environment this repository is developed against.

```bash
conda create -n hamlet python=3.11 && conda activate hamlet && pip install -e .
```

```bash
python3.11 -m venv .venv && . .venv/bin/activate && pip install -e .
```

The `env.yml` that used to be here is gone. It was a second dependency list alongside the code,
and a version it failed to pin is what once made `import hamlet` fail outright on a fresh install.
</details>

#### Point your IDE at it
	- Open the repository in your IDE.
	- Select the interpreter at `.venv/` in the repository root. In PyCharm:
	  `File->Settings->Project->Python Interpreter->Add->Existing environment`.

#### Optional: install a commercial solver (e.g. Gurobi)
Skip this step unless you need it — the examples run on HiGHS, which is already installed. To use
Gurobi instead, install it as below and set `solver: gurobi` under `optimization` in the
scenario's `agents.yaml`.

	- Go to gurobi.com
	- Create an account with your university email 
	- When the account has been activated, log in and download the newest Gurobi solver.
	- Go to Academia->Academic Program and Licenses
	- Follow the installation instructions under "Individual Academic Licenses" to activate your copy of Gurobi

### Test your installation
    - Navigate to ./PycharmProjects/hamlet/examples
	- Choose `create_simple_scenario`, which needs no solver licence (the other examples are configured for Gurobi)
    - Run the jupyter notebook `run.ipynb`
    - If everything is installed correctly, the notebook should run without errors and you should see the results of the example scenario.

## Citing HAMLET
If you use HAMLET in your research, please cite the following publication:

- M. Doepfert, J. Chu, T. Hamacher, [HAMLET: A modular agent-based Python framework for energy markets and
systems](https://www.sciencedirect.com/science/article/pii/S2352711025003127), 2025, [SoftwareX](https://www.sciencedirect.com/journal/softwarex), Volume 32, 12346, ISSN 2352-7110, [DOI:10.1016/j.softx.2025.102346](https://doi.org/10.1016/j.softx.2025.102346).

Please use the following BibTeX:

```
@article{HAMLET,
  author={Doepfert, Markus and Chu, Jiahe and Hamacher, Thomas},
  title={HAMLET: A modular agent-based Python framework for energy markets and systems},
  journal={SoftwareX},
  volume={32},
  pages={102346},
  year={2025},
  issn={2352-7110},
  doi={10.1016/j.softx.2025.102346},
  url={https://doi.org/10.1016/j.softx.2025.102346},
}
```

## Contact
Feel free to contact us if you want to contribute to this project, cooperate on an interesting research question
or just to ask about the project.

[Markus Doepfert](https://campus.tum.de/tumonline/ee/ui/ca2/app/desktop/#/pl/ui/$ctx/visitenkarte.show_vcard?$ctx=design=ca2;header=max;lang=de&pPersonenGruppe=3&pPersonenId=99801BCF1F13B4C9)  
markus.doepfert@tum.de  
Research Associate @ TUM ENS

[//]: # (## References)

[//]: # ()
[//]: # (TBD)


