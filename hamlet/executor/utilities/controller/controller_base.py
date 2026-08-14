__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# Used to ensure a consistent design of all markets


def derive_energy_types(mapping: dict) -> list:
    """The unique energy types of a component mapping, in a stable order.

    Sorted, and a list rather than a set, because every backend iterates this to add one balance
    constraint and one slack variable pair per energy type -- so it fixes the row and column order
    of the agent's optimisation model. A set's iteration order depends on Python's per-process
    string hash seed, which permuted those orders from run to run, `update_socs` carried the
    difference into the next timestep, and two identical runs produced different results
    (issue #216, same mechanism as #198). The usual account of the middle step -- that the
    permuted model is degenerate enough for the solver to return a different equally-optimal
    vertex -- is the interpretation; what was measured is that sorting this set removes the
    divergence.

    Everything downstream tests membership or builds a `str.startswith` tuple, so the order itself
    is not depended upon -- only its stability across processes is.
    """
    return sorted({energy_type for component in mapping.values() for energy_type in component})


class ControllerBase:

    def __init__(self):
        pass

    def run(self):
        raise NotImplementedError()
