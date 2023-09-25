# This file contains all constants used in the project


# KEYS
K_ACCOUNT = 'account'
K_PLANTS = 'plants'
K_EMS = 'ems'
K_MARKET = 'market'
K_FORECASTS = 'forecasts'
K_TIMESERIES = 'timeseries'
K_SETPOINTS = 'setpoints'
K_TARGET = 'target'     # relevant for forecast train data
K_FEATURES = 'features'     # relevant for forecast train data


# UNIT CONSTANTS
WH_TO_MWH = 1e-6
MWH_TO_WH = 1e6
WH_TO_KWH = 1e-3
KWH_TO_WH = 1e3
PERCENT_TO_FRACTION = 1e-2
FRACTION_TO_PERCENT = 1e2
E5_TO_FRACTION = 1e-5
FRACTION_TO_E5 = 1e5
PERCENT_TO_E5 = 1e-3
E5_TO_PERCENT = 1e3


# TIME CONSTANTS
SECONDS_TO_HOURS = 1 / 3600
HOURS_TO_SECONDS = 3600
SECONDS_TO_MINUTES = 1 / 60
MINUTES_TO_SECONDS = 60
MINUTES_TO_HOURS = 1 / 60
HOURS_TO_MINUTES = 60
MINUTES_TO_DAYS = 1 / 1440
DAYS_TO_MINUTES = 1440
SECONDS_TO_DAYS = 1 / 86400
DAYS_TO_SECONDS = 86400
HOURS_TO_DAYS = 1 / 24
DAYS_TO_HOURS = 24

# ENERGY TYPES
ET_ELECTRICITY = 'power'
ET_HEAT = 'heat'
ET_COOLING = 'cold'
ET_H2 = 'h2'

# OPERATION MODES
# Note: Storage is not an operation mode. They are always modeled as loads and have negative values when generating.
#       This can be changed for every controller individually though as it is only a convention.
OM_GENERATION = 'gen'
OM_LOAD = 'load'
OM_STORAGE = 'storage'


### TABLES ###
# NAMES
TN_TIMETABLE = 'timetable'

# COLUMNS
TC_TIMESTAMP = 'timestamp'
TC_TIMESTEP = 'timestep'
TC_REGION = 'region'
TC_MARKET = 'market'
TC_NAME = 'name'
TC_ACTION = 'action'
TC_TYPE = 'type'
TC_METHOD = 'method'
TC_COUPLING = 'coupling'
TC_TYPE_TRANSACTION = 'type_transaction'
TC_ID_AGENT = 'id_agent'
TC_ID_AGENT_IN = 'id_agent_in'
TC_ID_AGENT_OUT = 'id_agent_out'
TC_ID_METER = 'id_meter'
TC_ENERGY = 'energy'
TC_ENERGY_IN = 'energy_in'
TC_ENERGY_OUT = 'energy_out'
TC_ENERGY_USED = 'energy_used'
TC_PRICE_PU = 'price_pu'
TC_PRICE_PU_IN = 'price_pu_in'
TC_PRICE_PU_OUT = 'price_pu_out'
TC_PRICE = 'price'
TC_PRICE_IN = 'price_in'
TC_PRICE_OUT = 'price_out'
TC_POWER = 'power'
TC_POWER_IN = 'power_in'
TC_POWER_OUT = 'power_out'
TC_BALANCE_ACCOUNT = 'balance_account'
TC_QUALITY = 'quality'
TC_SHARE_QUALITY = 'share_quality'
TC_TYPE_METER = 'type_meter'
TC_TYPE_PLANTS = 'type_plants'
TC_SOC = 'soc'
TC_PLANT_VALUE = 'plant_value'

# columns related to weather
TC_CLOUD_COVER = 'cloud_cover'
TC_TEMPERATURE = 'temp'
TC_TEMPERATURE_FEELS_LIKE = 'temp_feels_like'
TC_TEMPERATURE_MIN = 'temp_min'
TC_TEMPERATURE_MAX = 'temp_max'
TC_PRESSURE = 'pressure'
TC_HUMIDITY = 'humidity'
TC_VISIBILITY = 'visibility'
TC_WIND_SPEED = 'wind_speed'
TC_WIND_DIRECTION = 'wind_dir'
TC_SUN_RISE = 'sunrise'
TC_SUN_SET = 'sunset'
TC_POP = 'pop'
TC_GHI = 'ghi'
TC_DHI = 'dhi'
TC_DNI = 'dni'

# PLANTS
P_INFLEXIBLE_LOAD = 'inflexible_load'
P_FLEXIBLE_LOAD = 'flexible_load'
P_HEAT = 'heat'
P_DHW = 'dhw'
P_PV = 'pv'
P_WIND = 'wind'
P_FIXED_GEN = 'fixed_gen'
P_HP = 'hp'
P_EV = 'ev'
P_BATTERY = 'battery'
P_PSH = 'psh'
P_HYDROGEN = 'hydrogen'
P_HEAT_STORAGE = 'heat_storage'