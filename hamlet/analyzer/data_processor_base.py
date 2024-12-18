class DataProcessorBase:
    def __init__(self, path: dict, config: dict):
        # Path storing relevant results
        self.path = path

        # Configuration dictionary
        self.config = config

        # Processed data
        self.data = {}

    def process(self, **kwargs):
        """Process all relevant results data."""
        # get all process data functions, function has to start with 'process_' to be included
        process_functions = [func for func in dir(self) if callable(getattr(self, func)) and 'process_' in func]

        # iterate through all functions and add returns from functions to self.data
        for func in process_functions:
            self.data[func.replace('process_', '')] = getattr(self, func)(**kwargs)

        return self.data
