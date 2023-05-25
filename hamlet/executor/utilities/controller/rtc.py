class Rtc:

    def __init__(self, **kwargs):
        self._rtc(**kwargs)


    def _rtc(self, **kwargs):

        # Obtain data
        data = kwargs['data']
        account = data['account']
        plants = data['plants']
        meters = data['meters']
        socs = data['socs']
        timeseries = data['timeseries']
        forecasts = data['forecasts']
        setpoints = data['setpoints']
        timetable = kwargs['timetable']
        market = kwargs['market']

        # Calculate delta_t (in seconds)
        delta_t = (timetable['timestep'][1] - timetable['timestep'][0]).total_seconds()

        # Set all balancing to zero
        bal_elec = 0
        bal_heat = 0
        bal_dhw = 0
        bal_cool = 0  # not implemented yet
        bal_gas = 0  # not implemented yet

        # Aggregate all demands for each type
        # Electricity
        for demand in self.__get_plants(plants=plants, plant_types=['inflexible_load, flexible_load']):
            bal_elec += timeseries[demand]  # TODO: Find correct column (that contains demand ID)
            setpoints[demand] = timeseries[demand]
        # Heat
        for demand in self.__get_plants(plants=plants, plant_types='heat'):
            bal_heat += timeseries[demand]  # TODO: Find correct column (that contains demand ID)
            setpoints[demand] = timeseries[demand]
        # Drinking hot water
        for demand in self.__get_plants(plants=plants, plant_types='dhw'):
            bal_dhw += timeseries[demand]  # TODO: Find correct column (that contains demand ID)
            setpoints[demand] = timeseries[demand]
        # Cool
        bal_cool = 0
        # Gas
        bal_gas = 0

        # Aggregate all generation for each type
        # Electricity
        for generation in self.__get_plants(plants=plants, plant_types=['pv', 'wind', 'fixedgen']):
            bal_elec += timeseries[generation]  # TODO: Find correct column (that contains demand ID)
            setpoints[demand] = timeseries[generation]

        # Special methods
        # Add EVs to electricity demand
        for ev in self.__get_plants(plants=plants, plant_types='ev'):

            # Get availability
            availability = timeseries[f'{ev}_availability']

            # If not at home, do nothing
            if availability == 0:
                setpoints[ev] = 0
                continue

            # Get current SoC
            soc = socs[ev]

            # If SoC is already at 100%, do nothing
            # Note: The SoC is measured from 0 to 1e5 which is 0 to 100%
            if soc == 1e5:
                setpoints[ev] = 0
                continue

            # If SoC is below 100%, charge

            # Calculate the amount of energy that can be charged
            energy_to_charge = plants[ev]['capacity'] * (1e5 - soc) / 1e5

            # Calculate power needed to charge the EV to full within the time step
            power = energy_to_charge / delta_t

            # Limit power to the maximum power of the EV
            power = (min(power, plants[ev]['charging_home'])).astype('int32')

            # Update SoC
            socs[ev] = (soc + power * delta_t / plants[ev]['capacity'] * 1e5).astype('int16')

            # Set the setpoint (negative as it is a demand)
            setpoints[ev] = power * -1

            # Add demand to balance (negative as it is a demand)
            bal_elec = power * -1

        # Add HPs to electricity demand and heat generation
        # TODO: Considerations how to handle multiple HPs
        for hp in self.__get_plants(plants=plants, plant_types='hp'):

            # Check if power is sufficient to cover heat demand
            if - bal_heat < plants[hp]['power']:
                # If so, set power to heating demand (positive as it is generation)
                setpoints[hp] = bal_heat * - 1 * timeseries[hp]  # TODO: Choose correct column (heat)
                # Set heat balance to zero
                bal_heat = 0

            # TODO: Next step would be to check if also DHW demand can be covered
            power_th = plants[hp]['power'] - setpoints[hp]   # TODO: Choose correct column (dhw)



        return data

    def _controller_rtc_max_pv(self, **kwargs):

        # Obtain data
        data = kwargs['data']
        timetable = kwargs['timetable']
        market = kwargs['market']

        # Call standard rtc
        data = self._controller_rtc(data=data, timetable=timetable, market=market)

        # Adjust to use max pv

        return data
