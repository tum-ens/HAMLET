class PlotterBase:
    def __init__(self, path: dict, config: dict, data: dict):
        # Path storing relevant results
        self.path = path

        # Configuration dictionary
        self.config = config

        # Processed results data
        self.data = data

        # Plotted figures
        self.figures = {}

    def plot(self, **kwargs):
        """Plot all relevant results data."""
        # get all plot data functions, function has to start with 'plot_' to be included
        plot_functions = [func for func in dir(self) if callable(getattr(self, func)) and 'plot_' in func]

        # iterate through all functions and plot results
        for func in plot_functions:
            self.figures[func.replace('plot_', '')] = getattr(self, func)(**kwargs)

        return self.figures
