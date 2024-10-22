__author__ = "HodaHamdy"
__credits__ = "MarkusDoepfert"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import polars as pl

import hamlet.constants as c


class RtcBase:
    def __init__(self, **kwargs):
        # Create the model
        self.model = self.get_model(**kwargs)

        # Store the mapping of the components to the energy types and operation modes
        self.mapping = kwargs['mapping']
        # Identify all unique energy types
        self.energy_types = set()
        for mapping in self.mapping.values():
            self.energy_types.update(mapping.keys())

        # Get the timetable and filter it to only include the rows with the current timestep
        self.timetable = kwargs[c.TN_TIMETABLE]
        # Get the delta between timestamps
        self.dt = self.timetable[1, c.TC_TIMESTEP] - self.timetable[0, c.TC_TIMESTEP]
        # Filter the timetable to only include the rows with the current timestamp
        self.timetable = self.timetable.filter(pl.col(c.TC_TIMESTAMP) == pl.col(c.TC_TIMESTEP))
        # Get the current timestamp
        self.timestamp = self.timetable[0, c.TC_TIMESTAMP]

        # Get the agent and other data
        self.agent = kwargs['agent']
        self.account = self.agent.account
        self.plants = self.agent.plants  # Formerly known as components
        self.setpoints = self.agent.setpoints
        self.timeseries = self.agent.timeseries
        self.socs = self.agent.socs
        self.meters = self.agent.meters
        # Filter the timeseries to only include the rows with the current timestamp
        self.timeseries = self.timeseries.join(self.timetable, on=c.TC_TIMESTAMP, how='semi')
        # Get the targets by filtering the setpoints to only include the rows with the current timestamp
        self.targets = self.setpoints.join(self.timetable, on=c.TC_TIMESTAMP, how='semi')

        # Raise warning if timeseries exceeds one row
        if len(self.timeseries) != 1:
            raise ValueError(f"Timeseries has {len(self.timeseries)} rows. It should only have 1 row for the rtc.")

        # Get the market data
        self.market = kwargs[c.TC_MARKET]
        # Get market name
        # Get the market names and types
        self.market_names = self.timetable.select(c.TC_NAME).unique().to_series().to_list()
        self.market_types = self.timetable.select(c.TC_MARKET).unique().to_series().to_list()
        # Assign each market name to an energy type
        self.markets = {name: c.TRADED_ENERGY[mtype] for name, mtype in zip(self.market_names, self.market_types)}
        # Get the market results for each market
        self.market_results = {}
        for market_type, market in self.market.items():
            for market_name, data in market.items():
                # Get transactions table
                transactions = data.market_transactions
                # Filter for agent ID
                transactions = transactions.filter(pl.col(c.TC_ID_AGENT) == self.agent.agent_id)
                # Filter for current timestamp
                transactions = transactions.filter(pl.col(c.TC_TIMESTEP) == self.timestamp)
                # Fill NaN values with 0
                transactions = transactions.fill_null(0)
                # Get net energy amount for market
                self.market_results[market_name] = (transactions
                .select(pl.sum(c.TC_ENERGY_IN).cast(pl.Int64)
                        - pl.sum(c.TC_ENERGY_OUT).cast(pl.Int64))
                .to_series().to_list()[0])

        # Obtain maximum balancing power
        # TODO: Deprecate balancing market in the equations. The relevant value is the market itself.
        #  Balancing occurs in the market section of the code.
        # TODO: self.balancing = db.get_balancing_power(market name, energy type)

        # Available plants
        self.available_plants = self.get_available_plants()

        # Note: This can probably be only done once and then stored in the agent. Afterwards, it only needs to be
        #  updated every timestep (will need considerable adjustment though and might not be worth the effort).

        # Create the plant objects
        self.plant_objects = {}
        self.create_plants()

        # Create the market objects
        self.market_class = self.get_market_class()
        self.market_objects = {}
        self.create_markets()

        # Define the model
        self.define_variables()
        self.define_constraints()
        self.define_objective()

    def run(self):
        raise NotImplementedError()

    def create_plants(self):
        """
        Create the plant objects for the optimization problem
        """
        for plant_name, plant_data in self.plants.items():

            # Get the plant type from the plant data
            plant_type = plant_data['type']

            # Retrieve the timeseries data for the plant
            cols = [col for col in self.timeseries.columns if col.startswith(plant_name)]
            timeseries = self.timeseries.select(cols)

            # Retrieve the target setpoints for the plant
            cols = [col for col in self.targets.columns if col.startswith(plant_name)]
            targets = self.targets.select(cols)

            # Retrieve the soc data for the plant (if applicable)
            cols = [col for col in self.socs.columns if col.startswith(plant_name)]
            socs = self.socs.select(cols)

            # Get the plant class
            plant_class = self.available_plants.get(plant_type)
            if plant_class is None:
                raise ValueError(f"Unsupported plant type: {plant_name}")

            # Create the plant object
            self.plant_objects[plant_name] = plant_class(name=plant_name, timeseries=timeseries, **plant_data,
                                                         targets=targets, socs=socs, delta=self.dt)

        return self.plant_objects

    def create_markets(self):
        """
        Create the market objects for the optimization problem
        """

        # Define variables from the market results and a balancing variable for each energy type
        for market in self.markets:
            # Create market object
            self.market_objects[market] = self.market_class(name=market,
                                                            timeseries=self.market,
                                                            market_result=self.market_results[market],
                                                            delta=self.dt)

        return self.market_objects

    def process_solution(self):

        # Obtain the solution values
        solution = self.get_solution()

        # Update setpoints
        self.setpoints = self.update_setpoints(solution)

        # Update socs
        self.socs = self.update_socs(solution)

        # Update meters
        self.meters = self.update_meters(solution)

        # Update the agent
        self.agent.setpoints = self.setpoints
        self.agent.socs = self.socs
        self.agent.meters = self.meters

        return self.agent

    def update_setpoints(self, solution: dict) -> pl.DataFrame:

        # Set all setpoints to 0
        self.setpoints = self.setpoints.select([pl.col(self.setpoints.columns[0])] +
                                               [pl.lit(0).alias(col) for col in self.setpoints.columns[1:]]
                                               )

        # Get relevant column name beginnings (i.e. the plant names and market and balancing)
        beginnings = set([col.split('_', 1)[0] for col in solution.keys()
                          if not col.startswith('objective') and not col.startswith('balance')])
        # Get relevant column name endings (i.e. the energy types)
        endings = self.energy_types
        # Get the relevant columns
        src_cols = [col for col in solution.keys()
                    if col.split('_', 1)[0] in beginnings and col.rsplit('_', 1)[-1] in endings]

        # Shift index according to timetable time
        timesteps = [self.timetable[0, c.TC_TIMESTAMP] + self.dt * t for t in range(len(self.setpoints))]
        self.setpoints = self.setpoints.with_columns(pl.Series(timesteps)
                                                     .cast(pl.Datetime(time_unit='ns', time_zone='UTC'))
                                                     .alias(c.TC_TIMESTAMP))

        # Update setpoints
        # TODO: Do this similar to the one in mpc (columns are added if missing, otherwise replaced)
        for src_col in src_cols:
            # Check if the column is already in the setpoints
            if src_col not in self.setpoints.columns:
                # Add column to dataframe with 0 values
                self.setpoints = self.setpoints.with_columns(pl.lit(0).alias(src_col))
            # Assign setpoint value to first row
            self.setpoints[0, src_col] = round(solution[src_col])

        # Sum the respective market columns into one column
        # (Will be deprecated once the balancing variable is taken out of the equations)
        self.setpoints, src_cols = self.sum_market_columns(self.setpoints, src_cols, list(self.markets.keys()))

        # Drop all setpoint columns that are not part of src_cols (plus keep timestamp and timestep column)
        sel_cols = [self.setpoints.columns[0]] + src_cols
        self.setpoints = self.setpoints.select(sel_cols)

        return self.setpoints

    def update_socs(self, solution: dict) -> pl.DataFrame:

        # Find row that is to be updated (i.e. the row with the next timestamp)
        row_soc = self.socs.filter(self.socs[c.TC_TIMESTAMP] == self.timestamp + self.dt)

        # Update socs
        for col in self.socs.columns[1:]:
            # Extract power from variable values
            key = next((key for key in solution
                        if key.startswith(col) and (key.endswith('_power') or key.endswith('_heat'))),
                       None)

            if key:  # Check for matching key
                # Get power from solution
                # Note: Since negative power means that the plant takes energy from the grid,
                #  we need to multiply the power by -1. Thus the signs are also reversed subsequently.
                power = solution[key] * -1

                # Get the column dtype
                dtype = self.socs[col].dtype

                # Get soc from plant object
                soc = self.plant_objects[col].soc

                # Calculate delta for the soc based on power and time step
                delta_soc = power * self.dt.total_seconds() * c.SECONDS_TO_HOURS

                # Adjust delta_soc by efficiency based on power being positive or negative
                if power > 0:
                    delta_soc *= self.plant_objects[col].efficiency
                elif power < 0:
                    delta_soc /= self.plant_objects[col].efficiency

                # Update soc
                soc += delta_soc

                # Round soc to integer
                soc = round(soc)

                # Update the soc value in the DataFrame for the corresponding column
                row_soc = row_soc.with_columns(pl.lit(soc).cast(dtype).alias(col))

            # Ensure that soc is within bounds (not implemented as of now to ensure that the model is working)
            # soc = max(0, min(self.plant_objects[col].capacity, soc))

        # Update socs dataframe
        self.socs = self.socs.filter(self.socs[c.TC_TIMESTAMP] != self.timestamp + self.dt)
        self.socs = self.socs.merge_sorted(row_soc, key=c.TC_TIMESTAMP)

        return self.socs

    def update_meters(self, solution: dict) -> pl.DataFrame:

        # Find row that is to be updated as well as the previous row (i.e. current and next timestamp)
        row_now = self.meters.filter(self.meters[c.TC_TIMESTAMP] == self.timestamp)
        row_new = self.meters.filter(self.meters[c.TC_TIMESTAMP] == self.timestamp + self.dt)

        # Create strings for energy types
        energy_endings = tuple(f'_{et}' for et in self.energy_types)

        # Update meters
        for col in self.meters.columns[1:]:
            # Extract power from variable values
            key = next((key for key in solution if key.startswith(col) and key.endswith(energy_endings)), None)

            if key:  # Check for matching key
                # Calculate energy from power
                delta_energy = solution[key] * self.dt.total_seconds() * c.SECONDS_TO_HOURS

                # Get the column dtype
                dtype = self.meters[col].dtype

                # Get old meter value from row now
                meter_now = row_now[col]

                # Update meter value in row new
                row_new = row_new.with_columns(pl.lit(meter_now + round(delta_energy)).cast(dtype).alias(col))

        # Update meters dataframe
        self.meters = self.meters.filter(self.meters[c.TC_TIMESTAMP] != self.timestamp + self.dt)
        self.meters = self.meters.merge_sorted(row_new, key=c.TC_TIMESTAMP)

        return self.meters

    @staticmethod
    def sum_market_columns(df: pl.DataFrame, src_cols: list, markets: list):

        # Loop through each market
        for market_name in markets:
            # Get the list of columns starting with 'lem_conti'
            columns_to_combine = [col for col in df.columns if col.startswith(market_name)]

            # Find the column with the shortest length
            col_name = columns_to_combine[columns_to_combine.index(min(columns_to_combine, key=len))]

            # Combine columns into a new column
            df = df.with_columns(
                df.select(columns_to_combine).sum(axis=1).alias(col_name),
            )

        # Drop the columns containing 'balancing'
        src_cols = [col for col in src_cols if c.MT_BALANCING not in col]

        return df, src_cols

    def get_model(self, **kwargs):
        raise NotImplementedError

    def get_available_plants(self):
        raise NotImplementedError

    def get_market_class(self):
        raise NotImplementedError

    def get_solution(self):
        raise NotImplementedError

    def define_variables(self):
        pass

    def define_constraints(self):
        pass

    def define_objective(self):
        pass
