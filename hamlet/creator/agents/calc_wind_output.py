import pandas as pd
from windpowerlib import ModelChain, WindTurbine


def get_weather_data(weather_path):
    """adjust weather data to the right format that windpowerlib needed.

    Args:
        weather_path: path of the original weather data (string)

    Returns:
        weather: adjusted weather data (dataframe)

    """
    # get weather data from csv
    weather = pd.read_csv(weather_path)
    weather = weather[weather['ts_delivery_current'] == weather['ts_delivery_fcast']]  # remove forcasting data

    # convert time data to datetime
    time = pd.DatetimeIndex(pd.to_datetime(weather['ts_delivery_current'], unit='s', utc=True))
    weather.index = time.tz_convert("Europe/Berlin")
    weather.index.name = None

    # delete unnecessary columns and rename
    weather.drop(['ts_delivery_fcast', 'cloud_cover', 'sunrise', 'sunset', 'ghi', 'dhi',
                  'visibility', 'pop'], axis=1, inplace=True)
    weather.rename(columns={'temp': 'temperature'}, inplace=True)

    if 'roughness_length' not in weather.columns:
        weather['roughness_length'] = 0.15

    # generate height level hard-coded
    weather.columns = pd.MultiIndex.from_tuples(tuple(zip(weather.columns, [2, 2, 2, 2, 2, 2, 2, 10, 10, 2])),
                                                names=('', 'height'))

    return weather


def calc_wind_output_from_spec(weather_path, turbine):
    """Calculate wind power output for the given weather and turbine config.

     Args:
         weather_path: path of weather data (string)
         turbine: turbine data (dic)

     Returns:
         power: calculated wind output power pu from turbine model chain (Dataframe)

     """
    # get weather data
    weather = get_weather_data(weather_path)

    # get nominal power
    nominal_power = turbine['nominal_power']

    # convert power curve to dataframe
    turbine['power_curve'] = pd.DataFrame(data={
                "value": turbine['power_curve'],
                "wind_speed": turbine['wind_speed']})

    # convert power coefficient curve to dataframe
    turbine['power_coefficient_curve'] = pd.DataFrame(data={
        "value": turbine['power_coefficient_curve'],
        "wind_speed": turbine['wind_speed']})

    # generate a WindTurbine object from data
    turbine = WindTurbine(**turbine)

    # calculate turbine model
    mc_turbine = ModelChain(turbine).run_model(weather)

    # get output power
    power = mc_turbine.power_output

    # set time index to origin timestamp
    power.index = weather['ts_delivery_current'].unstack(level=0).values
    power.index.name = 'timestamp'

    # calculate power pu related to nominal power
    power = power / nominal_power

    # rename data column
    power.rename('power', inplace=True)
    power = power.to_frame()

    return power
