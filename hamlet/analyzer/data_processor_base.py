import os


class DataProcessorBase:
    def __init__(self, path: dict, config: dict, name_subdirectory: str):
        # Configuration dictionary
        self.config = config

        # Processed data
        self.data = {}

        # subdirectory name
        self.name_subdirectory = name_subdirectory

        # Combine normal path and subdirectory path
        self.path = {}
        for path_key, value in path.items():
            self.path[path_key] = os.path.join(value, self.name_subdirectory)

    def process(self, **kwargs):
        """Process all relevant results data."""
        # get all process data functions, function has to start with 'process_' to be included
        process_functions = [func for func in dir(self) if callable(getattr(self, func)) and 'process_' in func]

        # iterate through all functions and add returns from functions to self.data
        for func in process_functions:
            self.data[func.replace('process_', '')] = getattr(self, func)(**kwargs)

        return self.data
