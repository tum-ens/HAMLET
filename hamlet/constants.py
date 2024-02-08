# This file contains all constants used in the project
import polars as pl

# KEYS
K_GENERAL = 'general'
K_ACCOUNT = 'account'
K_PLANTS = 'plants'
K_EMS = 'ems'
K_MARKET = 'market'
K_FORECASTS = 'forecasts'
K_TIMESERIES = 'timeseries'
K_SETPOINTS = 'setpoints'
K_TARGET = 'target'  # relevant for forecast train data
K_FEATURES = 'features'  # relevant for forecast train data

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
COP_TO_COP100 = 100
COP100_TO_COP = 1 / 100

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

# MONEY CONSTANTS
EUR_TO_CENT = 100
CENT_TO_EUR = 1 / 100
EUR_TO_EURe7 = 1e7
EURe7_TO_EUR = 1e-7
CENT_TO_EURe7 = 1e9
EURe7_TO_CENT = 1e-9

# WEATHER CONSTANTS
KELVIN_TO_CELSIUS = -273.15
CELSIUS_TO_KELVIN = 273.15

# OTHER CONSTANTS
EUR_KWH_TO_EURe7_WH = EUR_TO_EURe7 / KWH_TO_WH  # conversion to ensure that the values are integers

# ENERGY TYPES
ET_ELECTRICITY = 'power'
ET_HEAT = 'heat'
ET_COOLING = 'cold'
ET_H2 = 'h2'

# SYMBOLS (symbols used for the units in the tables)
S_POWER = 'P'
S_ENERGY = 'E'
S_PRICE = 'price'
S_PRICE_PU = 'price_pu'
S_SOC = 'soc'
S_COP = 'COP100'

# MARKET TYPES
MT_LEM = 'lem'
MT_LFM = 'lfm'
MT_LHM = 'lhm'
MT_LCM = 'lcm'
MT_LH2M = 'lh2m'
MT_WHOLESALE = 'wholesale'  # might not be needed
MT_BALANCING = 'balancing'  # might not be needed

# MARKET ACTIONS
MA_CLEAR = 'clear'
MA_SETTLE = 'settle'

# MARKET CLEARING TYPES
MCT_EX_ANTE = 'ex-ante'
MCT_EX_POST = 'ex-post'

# MARKET CLEARING METHODS
MCM_PDA = 'pda'
MCM_COMMUNITY = 'community'

# MARKET PRICING
MP_UNIFORM = 'uniform'
MP_DISCRIMINATORY = 'discriminatory'

# MARKET COUPLING
MC_ABOVE = 'above'
MC_BELOW = 'below'

# TRADE TYPES
TT_MARKET = 'market'
TT_RETAIL = 'retail'
TT_GRID = 'grid'
TT_LEVIES = 'levies'
TT_BALANCING = 'balancing'

# TRADED ENERGY TYPES
TRADED_ENERGY = {
    MT_LEM: ET_ELECTRICITY,
    MT_LFM: ET_ELECTRICITY,
    MT_LHM: ET_HEAT,
    MT_LCM: ET_COOLING,
    MT_LH2M: ET_H2,
}

# OPERATION MODES
# Note: Storage is not an operation mode. They are modeled as loads and have negative values when generating.
#       This can be changed for every controller individually though as it is only a convention.
OM_GENERATION = 'gen'
OM_LOAD = 'load'
OM_STORAGE = 'storage'

# POWER FLOWS
PF_IN = 'in'
PF_OUT = 'out'

### TABLES ###
# NAMES
TN_TIMETABLE = 'timetable'
TN_MARKET_TRANSACTIONS = 'market_transactions'
TN_BIDS_CLEARED = 'bids_cleared'
TN_BIDS_UNCLEARED = 'bids_uncleared'
TN_OFFERS_CLEARED = 'offers_cleared'
TN_OFFERS_UNCLEARED = 'offers_uncleared'
TN_POSITIONS_MATCHED = 'positions_matched'

# COLUMNS
TC_TIMESTAMP = 'timestamp'
TC_TIMESTEP = 'timestep'
TC_REGION = 'region'
TC_MARKET = 'market'
TC_NAME = 'name'
TC_ENERGY_TYPE = 'energy_type'
TC_ACTIONS = 'action'  # TODO: Change to actions
TC_CLEARING_TYPE = 'type'  # TODO: Change to clearing_type
TC_CLEARING_METHOD = 'method'  # TODO: Change to clearing_method
TC_CLEARING_PRICING = 'pricing'  # TODO: Change to clearing_pricing
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

# SCHEMAS
# Note: The schemas are used to define the data types of the columns in the tables and are taken from tables.xlsx
TS_MARKET_TRANSACTIONS = {TC_TIMESTAMP: pl.Datetime(time_unit='ns', time_zone='UTC'),
                          TC_TIMESTEP: pl.Datetime(time_unit='ns', time_zone='UTC'),
                          TC_REGION: pl.Categorical,
                          TC_MARKET: pl.Categorical,
                          TC_NAME: pl.Categorical,
                          TC_ENERGY_TYPE: pl.Categorical,
                          TC_TYPE_TRANSACTION: pl.Categorical,
                          TC_ID_AGENT: pl.Categorical,
                          TC_ENERGY_IN: pl.UInt64,
                          TC_ENERGY_OUT: pl.UInt64,
                          TC_PRICE_PU_IN: pl.Int32,
                          TC_PRICE_PU_OUT: pl.Int32,
                          TC_PRICE_IN: pl.Int64,
                          TC_PRICE_OUT: pl.Int64,
                          TC_QUALITY: pl.UInt8}
TS_BIDS_OFFERS = {TC_TIMESTAMP: pl.Datetime(time_unit='ns', time_zone='UTC'),
                  TC_TIMESTEP: pl.Datetime(time_unit='ns', time_zone='UTC'),
                  TC_REGION: pl.Categorical,
                  TC_MARKET: pl.Categorical,
                  TC_NAME: pl.Categorical,
                  TC_ENERGY_TYPE: pl.Categorical,
                  TC_ID_AGENT: pl.Categorical,
                  TC_ENERGY_IN: pl.UInt64,
                  TC_ENERGY_OUT: pl.UInt64,
                  TC_PRICE_PU_IN: pl.Int32,
                  TC_PRICE_PU_OUT: pl.Int32,
                  TC_PRICE_IN: pl.Int64,
                  TC_PRICE_OUT: pl.Int64,
                  TC_QUALITY: pl.UInt8}
TS_BIDS_CLEARED = {TC_TIMESTAMP: pl.Datetime(time_unit='ns', time_zone='UTC'),
                   TC_TIMESTEP: pl.Datetime(time_unit='ns', time_zone='UTC'),
                   TC_REGION: pl.Categorical,
                   TC_MARKET: pl.Categorical,
                   TC_NAME: pl.Categorical,
                   TC_ENERGY_TYPE: pl.Categorical,
                   TC_ID_AGENT_IN: pl.Categorical,
                   TC_ENERGY_IN: pl.UInt64,
                   TC_PRICE_PU_IN: pl.Int32,
                   TC_PRICE_IN: pl.Int64,
                   TC_QUALITY: pl.UInt8}
TS_BIDS_UNCLEARED = TS_BIDS_CLEARED
TS_OFFERS_CLEARED = {TC_TIMESTAMP: pl.Datetime(time_unit='ns', time_zone='UTC'),
                     TC_TIMESTEP: pl.Datetime(time_unit='ns', time_zone='UTC'),
                     TC_REGION: pl.Categorical,
                     TC_MARKET: pl.Categorical,
                     TC_NAME: pl.Categorical,
                     TC_ENERGY_TYPE: pl.Categorical,
                     TC_ID_AGENT_OUT: pl.Categorical,
                     TC_ENERGY_OUT: pl.UInt64,
                     TC_PRICE_PU_OUT: pl.Int32,
                     TC_PRICE_OUT: pl.Int64,
                     TC_QUALITY: pl.UInt8}
TS_OFFERS_UNCLEARED = TS_OFFERS_CLEARED
TS_POSITIONS_MATCHED = {TC_TIMESTAMP: pl.Datetime(time_unit='ns', time_zone='UTC'),
                        TC_TIMESTEP: pl.Datetime(time_unit='ns', time_zone='UTC'),
                        TC_REGION: pl.Categorical,
                        TC_MARKET: pl.Categorical,
                        TC_NAME: pl.Categorical,
                        TC_ENERGY_TYPE: pl.Categorical,
                        TC_ID_AGENT_IN: pl.Categorical,
                        TC_ID_AGENT_OUT: pl.Categorical,
                        TC_ENERGY: pl.UInt64,
                        TC_PRICE_PU: pl.Int32,
                        TC_PRICE: pl.Int64,
                        TC_QUALITY: pl.UInt8}

# AGENTS
A_SFH = 'sfh'
A_MFH = 'mfh'
A_CTSP = 'ctsp'
A_INDUSTRY = 'industry'
A_PRODUCER = 'producer'
A_STORAGE = 'storage'
A_AGGREGATOR = 'aggregator'

# PLANTS (no underscores allowed in the plant names)
P_INFLEXIBLE_LOAD = 'inflexible-load'
P_FLEXIBLE_LOAD = 'flexible-load'
P_HEAT = 'heat'
P_DHW = 'dhw'
P_PV = 'pv'
P_WIND = 'wind'
P_FIXED_GEN = 'fixed-gen'
P_HP = 'hp'
P_EV = 'ev'
P_BATTERY = 'battery'
P_PSH = 'psh'
P_HYDROGEN = 'hydrogen'
P_HEAT_STORAGE = 'heat-storage'

# CONTROLLERS
C_RTC = 'rtc'
C_MPC = 'mpc'

# COMPONENT MAPPING
# Note: Key states which type of plant is addressed and the value states which type of operation it has for the given
#       energy type
COMP_MAP = {
    # Electricity
    P_INFLEXIBLE_LOAD: {ET_ELECTRICITY: OM_LOAD},
    P_FLEXIBLE_LOAD: {ET_ELECTRICITY: OM_LOAD},
    P_PV: {ET_ELECTRICITY: OM_GENERATION},
    P_WIND: {ET_ELECTRICITY: OM_GENERATION},
    P_FIXED_GEN: {ET_ELECTRICITY: OM_GENERATION},
    P_EV: {ET_ELECTRICITY: OM_STORAGE},
    P_BATTERY: {ET_ELECTRICITY: OM_STORAGE},
    P_PSH: {ET_ELECTRICITY: OM_STORAGE},
    P_HYDROGEN: {ET_ELECTRICITY: OM_STORAGE},

    # Heat
    P_HEAT: {ET_HEAT: OM_LOAD},
    P_DHW: {ET_HEAT: OM_LOAD},
    P_HEAT_STORAGE: {ET_HEAT: OM_STORAGE},

    # Hybrid
    P_HP: {ET_ELECTRICITY: OM_LOAD, ET_HEAT: OM_GENERATION},
}
