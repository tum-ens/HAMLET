import pvlib
import pandas as pd
import numpy as np
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain


def create_system_from_config(config, orientation):
    """create PV system from hardware config file.

    Args:
        config: pv hardware configuration from json file (dic)
        orientation: pv orientation (surface tilt, surface azimuth) (tuple)

    Returns:
        system: a PV system abject (PVSystem)

    """
    # get pv orientation
    surface_tilt, surface_azimuth = orientation

    # get hardware data
    module = pd.Series(config['module'])
    inverter = pd.Series(config['inverter'])

    # set temperature model
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    # generate pv system
    system = PVSystem(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        module_parameters=module,
        inverter_parameters=inverter,
        temperature_model_parameters=temperature_model_parameters
    )

    return system


def adjust_weather_data(weather_path, location):
    """adjust weather data to the right format that pvlib needed.

    Args:
        weather_path: path of the original weather data (string)
        location: location of the weather data (latitude, longitude, name, altitude, timezone) (tuple)

    Returns:
        weather: adjusted weather data (dataframe)

    """
    # get location data
    latitude, longitude, name, altitude, timezone = location

    # get weather data from csv
    weather = pd.read_csv(weather_path)
    weather = weather[weather['ts_delivery_current'] == weather['ts_delivery_fcast']]   # remove forcasting data

    # convert time data to datetime (use utc time overall in pvlib)
    time = pd.DatetimeIndex(pd.to_datetime(weather['ts_delivery_current'], unit='s', utc=True))
    weather.index = time
    weather.index.name = 'utc_time'

    # adjust temperature data
    weather.rename(columns={'temp': 'temp_air'}, inplace=True)  # rename to pvlib format
    weather['temp_air'] -= 273.15   # convert unit to celsius

    # get solar position
    # test data find in https://www.suncalc.org/#/48.1364,11.5786,15/2022.02.15/16:21/1/3
    solpos = pvlib.solarposition.get_solarposition(
        time=time,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        temperature=weather['temp_air'],
        pressure=weather['pressure'],
    )

    # calculate dni with solar position
    weather.loc[:, 'dni'] = (weather['ghi'] - weather['dhi']) / np.cos(solpos['zenith'])

    return weather


def calc_pv_output(location, system, weather):
    """calculate power output of the PV system at given location under given weather conditions.

    Args:
        location: location of the pv system (latitude, longitude, name, altitude, timezone) (tuple)
        system: PV system abject (PVSystem)
        weather: weather data at this location (dataframe)

    Returns:
        power_out: output ac power of PV system (dataframe)

    """
    # get location data and create corresponding pvlib Location object
    latitude, longitude, name, altitude, timezone = location
    location = Location(
        latitude,
        longitude,
        name=name,
        altitude=altitude,
        tz=timezone,
    )

    # create calculation model for the given pv system and location
    mc = ModelChain(system, location)

    # calculate model under given weather data and get output ac power from it
    mc.run_model(weather)
    power_out = mc.results.ac

    return power_out


def calc_pv_output_from_spec(config, orientation, location, weather_path):
    """generate pv timeseries dataframe from hardware config file.

    Args:
        config: pv hardware configuration from json file (dic)
        orientation: pv orientation (surface tilt, surface azimuth) (tuple)
        location: location of the pv system (latitude, longitude, name, altitude, timezone) (tuple)
        weather_path: path of the original weather data (string)

    Returns:
        power: calculated pv output power

    """
    # calculate pv power output
    system = create_system_from_config(config, orientation)
    weather = adjust_weather_data(weather_path, location)
    power = calc_pv_output(location, system, weather)

    # set time index to origin timestamp
    power.index = weather['ts_delivery_current']
    power.index.name = 'timestamp'

    # rename and round data column
    power.rename('power', inplace=True)
    power = power.to_frame()
    power = power.round().astype(int)

    return power
