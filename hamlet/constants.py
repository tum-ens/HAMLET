__author__ = "MarkusDoepfert"
__credits__ = "jiahechu"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# This file contains all constants used in the project
import polars as pl

# SCENARIO FORMAT
# The on-disk layout of a generated scenario folder, NOT the version of HAMLET itself (that is
# `VERSION`). The two are deliberately independent: a release that changes no scenario file must
# not invalidate everybody's scenarios, and a change to the scenario format between releases must
# not go undetected because the release number happened not to move.
#
# The Creator stamps this into `general/general.json`; the Executor and the Analyzer refuse to
# read a scenario carrying anything else. A scenario with no stamp at all predates the versioning
# and is treated as incompatible -- see `functions.check_scenario_format`.
#
# WHEN TO BUMP IT
#   Bump when a scenario folder written by the current Creator would be misread by the previous
#   Executor, or vice versa. In practice that is one of two situations:
#
#   (a) The *shape* changed -- a file or a column was added, removed or renamed.
#       `tests/integration/creator/test_scenario_format_shape.py` fails and tells you so; this
#       case is mechanical, and it runs in the default test suite.
#   (b) The *meaning* changed while the shape stayed the same -- a column now holds something
#       different from what it held before. Nothing structural can see this. The signal is the
#       golden master moving because of a change to what the *Creator writes*, rather than to
#       what the Executor computes with it. Version 1 exists because of exactly such a change
#       (the retailer in/out convention), which is why this rule is written down rather than
#       left to the test.
#
#   Bumping means: change the number here, and commit the matching
#   `tests/integration/creator/format/scenario_format_v<N>.json`. Never edit a reference file
#   belonging to an older version -- those record what that version's scenarios looked like.
#
# HISTORY
#   1  Current. Retailer columns follow one convention: `_out` is the direction the agent pays
#      for. Scenarios generated before this are unstamped, and are rejected because running them
#      through the current Executor silently applies grid fees and levies to feed-in.
SCENARIO_FORMAT_VERSION = 1

# KEYS
K_SCENARIO_FORMAT_VERSION = 'scenario_format_version'  # key holding the stamp in general.json
K_GENERAL = 'general'
K_GRID = 'grids'
K_WEATHER = 'weather'
K_RETAILER = 'retailer'
K_TASKS = 'tasks'
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
ET_ELECTRICITY = 'electricity'
ET_HEAT = 'heat'
ET_COOLING = 'cold'
ET_H2 = 'hydrogen'
ET = {ET_ELECTRICITY, ET_HEAT, ET_COOLING, ET_H2}

# GRID TYPES
G_ELECTRICITY = ET_ELECTRICITY
G_HEAT = ET_HEAT
G_COOLING = ET_COOLING
G_H2 = ET_H2

# Key under which the grid stage records each element's horizon profile on a pandapower net, for
# the restriction stage to run the horizon from. It lives here rather than on either class because
# `grids/electricity.py` writes it and `grid_restrictions/enwg_14a.py` reads it, and those two
# already import each other through `grid_restriction.py` -- naming it on either side is a circular
# import. `pandapowerNet` is a dict, so the key rides along through `deepcopy`.
GRID_HORIZON_PROFILES = 'hamlet_horizon_profiles'

# SYMBOLS (symbols used for the units in the tables)
S_POWER = 'P'
S_ENERGY = 'E'
S_PRICE = 'price'
S_PRICE_PU = 'price_pu'
S_SOC = 'soc'
S_COP = 'COP100'

# MARKET TYPES
MT_ELECTRICITY = ET_ELECTRICITY
MT_FLEXIBILITY = 'flexibility'
MT_HEAT = ET_HEAT
MT_COLD = ET_COOLING
MT_H2 = ET_H2
MT_RETAIL = 'retail'  # might not be needed
MT_BALANCING = 'balancing'  # might not be needed

# MARKET ACTIONS
MA_CLEAR = 'clear'
MA_SETTLE = 'settle'

# MARKET CLEARING TYPES
MCT_EX_ANTE = 'ex-ante'
MCT_EX_POST = 'ex-post'

# MARKET CLEARING METHODS
MCM_NONE = 'None'
MCM_PDA = 'pda'
MCM_COMMUNITY = 'community'

# MARKET PRICING
MP_UNIFORM = 'uniform'
MP_DISCRIMINATORY = 'discriminatory'

# MARKET COUPLING
MC_ABOVE = 'above'
MC_BELOW = 'below'

# MARKET COMMODITY TYPES
MCT_ENERGY = 'energy'

# TRADE TYPES
TT_MARKET = 'market'
TT_RETAIL = 'retail'
TT_ENERGY = 'energy'
TT_POWER = 'power'
TT_GRID = 'grid'
TT_LEVIES = 'levies'
TT_BALANCING = 'balancing'

# TRADED ENERGY TYPES
TRADED_ENERGY = {
    MT_ELECTRICITY: ET_ELECTRICITY,
    MT_FLEXIBILITY: ET_ELECTRICITY,
    MT_HEAT: ET_HEAT,
    MT_COLD: ET_COOLING,
    MT_H2: ET_H2,
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
TC_TYPE_ENERGY = 'type_energy'
TC_ACTIONS = 'action'  # TODO: Change to actions
TC_CLEARING_TYPE = 'type'  # TODO: Change to clearing_type
TC_CLEARING_METHOD = 'method'  # TODO: Change to clearing_method
TC_CLEARING_PRICING = 'pricing'  # TODO: Change to clearing_pricing
TC_COUPLING = 'coupling'
TC_TYPE_TRANSACTION = 'type_transaction'
TC_ID_AGENT = 'id_agent'
TC_ID_AGENT_IN = f'{TC_ID_AGENT}_{PF_IN}'
TC_ID_AGENT_OUT = f'{TC_ID_AGENT}_{PF_OUT}'
TC_ID_PLANT = 'id_plant'
TC_ID_TRADE = 'id_trade'
TC_ENERGY = 'energy'
TC_ENERGY_IN = f'{TC_ENERGY}_{PF_IN}'
TC_ENERGY_OUT = f'{TC_ENERGY}_{PF_OUT}'
TC_ENERGY_USED = f'{TC_ID_AGENT}_used'
TC_PRICE_PU = 'price_pu'
TC_PRICE_PU_IN = f'{TC_PRICE_PU}_{PF_IN}'
TC_PRICE_PU_OUT = f'{TC_PRICE_PU}_{PF_OUT}'
TC_PRICE = 'price'
TC_PRICE_IN = f'{TC_PRICE}_{PF_IN}'
TC_PRICE_OUT = f'{TC_PRICE}_{PF_OUT}'
TC_POWER = 'power'
TC_POWER_IN = f'{TC_POWER}_{PF_IN}'
TC_POWER_OUT = f'{TC_POWER}_{PF_OUT}'
TC_BALANCE_ACCOUNT = 'balance_account'
TC_QUALITY = 'quality'
TC_SHARE_QUALITY = 'share_quality'
TC_QUANTITY = 'quantity'
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

# TABLE SCHEMAS
# Note: The schemas are used to define the data types of the columns in the tables and are taken from tables.xlsx
SCHEMA = {
    TC_TIMESTAMP: pl.Datetime(time_unit='ns', time_zone='UTC'),
    TC_TIMESTEP: pl.Datetime(time_unit='ns', time_zone='UTC'),
    TC_REGION: pl.Categorical,
    TC_MARKET: pl.Categorical,
    TC_NAME: pl.Categorical,
    TC_TYPE_ENERGY: pl.Categorical,
    TC_ACTIONS: pl.Categorical,
    TC_CLEARING_TYPE: pl.Categorical,
    TC_CLEARING_METHOD: pl.Categorical,
    TC_CLEARING_PRICING: pl.Categorical,
    TC_COUPLING: pl.Categorical,
    TC_TYPE_TRANSACTION: pl.Categorical,
    TC_ID_AGENT: pl.Categorical,
    TC_ID_AGENT_IN: pl.Categorical,
    TC_ID_AGENT_OUT: pl.Categorical,
    TC_ID_PLANT: pl.Categorical,
    TC_ID_TRADE: pl.String,
    TC_ENERGY: pl.Int64,
    TC_ENERGY_IN: pl.UInt64,
    TC_ENERGY_OUT: pl.UInt64,
    TC_ENERGY_USED: pl.UInt64,
    TC_PRICE_PU: pl.Int32,
    TC_PRICE_PU_IN: pl.Int32,
    TC_PRICE_PU_OUT: pl.Int32,
    TC_PRICE: pl.Int64,
    TC_PRICE_IN: pl.Int64,
    TC_PRICE_OUT: pl.Int64,
    TC_POWER: pl.Int32,
    TC_POWER_IN: pl.UInt32,
    TC_POWER_OUT: pl.UInt32,
    TC_BALANCE_ACCOUNT: pl.Int64,
    TC_QUALITY: pl.UInt8,
    TC_SHARE_QUALITY: pl.Int8,
    TC_QUANTITY: pl.UInt64,
    TC_TYPE_METER: pl.Categorical,
    TC_TYPE_PLANTS: pl.Categorical,
    TC_SOC: pl.UInt16,
}


def create_subschema(schema: dict, columns: list) -> dict:
    """
    Extract a subschema from the full schema.
    """
    return {col: schema[col] for col in columns if col in schema}


TS_MARKET_TRANSACTIONS = create_subschema(SCHEMA,
                                          [TC_TIMESTAMP, TC_TIMESTEP, TC_REGION, TC_MARKET, TC_NAME, TC_TYPE_ENERGY,
                                           TC_TYPE_TRANSACTION, TC_ID_AGENT, TC_ENERGY_IN, TC_ENERGY_OUT,
                                           TC_PRICE_PU_IN, TC_PRICE_PU_OUT, TC_PRICE_IN, TC_PRICE_OUT])
TS_BIDS_OFFERS = create_subschema(SCHEMA,
                                  [TC_TIMESTAMP, TC_TIMESTEP, TC_REGION, TC_MARKET, TC_NAME, TC_TYPE_ENERGY,
                                   TC_ID_AGENT, TC_ENERGY_IN, TC_ENERGY_OUT, TC_PRICE_PU_IN, TC_PRICE_PU_OUT,
                                   TC_PRICE_IN, TC_PRICE_OUT])
TS_BIDS_CLEARED = create_subschema(SCHEMA,
                                   [TC_TIMESTAMP, TC_TIMESTEP, TC_REGION, TC_MARKET, TC_NAME, TC_TYPE_ENERGY,
                                    TC_ID_AGENT_IN, TC_ENERGY_IN, TC_PRICE_PU_IN, TC_PRICE_IN])
TS_BIDS_UNCLEARED = TS_BIDS_CLEARED
TS_OFFERS_CLEARED = create_subschema(SCHEMA,
                                     [TC_TIMESTAMP, TC_TIMESTEP, TC_REGION, TC_MARKET, TC_NAME, TC_TYPE_ENERGY,
                                      TC_ID_AGENT_OUT, TC_ENERGY_OUT, TC_PRICE_PU_OUT, TC_PRICE_OUT])
TS_OFFERS_UNCLEARED = TS_OFFERS_CLEARED
TS_POSITIONS_MATCHED = create_subschema(SCHEMA,
                                        [TC_TIMESTAMP, TC_TIMESTEP, TC_REGION, TC_MARKET, TC_NAME, TC_TYPE_ENERGY,
                                         TC_ID_AGENT_IN, TC_ID_AGENT_OUT, TC_ENERGY, TC_PRICE_PU, TC_PRICE])

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
C_CONTROLLER = 'controller'
C_RTC = 'rtc'
C_FBC = 'fbc'
C_RB = 'rb'  # category of rtc and fbc: rule-based
C_OPTIM = 'optimization'  # category of rtc and fbc: optimization; parent class to poi and linopy
C_RL = 'rl'  # category of fbc: reinforcement learning
C_POI = 'poi'  # subcategory of mpc and optim: PyOptInterface package
C_LINOPY = 'linopy'  # subcategory of mpc and optim: Linopy package

# Section 14a EnWG (BNetzA BK6-22-300, eq. 2.1 in Chu 2024): a heat pump or room cooling system
# whose *grid connection* power exceeds 11 kW may be dimmed to 40 % of that power. Below the
# limit, the flat guaranteed minimum applies instead -- the same one EVs and batteries get.
ENWG14A_HP_SCALING_LIMIT = 11_000  # W
ENWG14A_HP_SCALING_FACTOR = 0.4

# MARKET TRADING STRATEGIES
MTS_ZI = 'zi'
MTS_LINEAR = 'linear'
MTS_RETAILER = 'retailer'

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

########################################################################################################################
# Controller limits and penalties
########################################################################################################################

# Key under which per-agent overrides may be given in the controller configuration
K_LIMITS = 'limits'
K_PENALTIES = 'penalties'

# Bounds on the real-time control optimisation variables (unit: W).
# None means unbounded. The defaults below reproduce the historical behaviour, in which the
# market variable and the heat-pump fallbacks are unbounded and the balancing power is set to a
# placeholder large enough never to bind.
# Override per agent through the `limits` block of the rtc controller configuration, e.g.
#   controller:
#     rtc:
#       limits:
#         balancing_power: 4000000
#         market_power: 4000000
# TODO: balancing_power should be derived from the retailer's declared balancing energy
#  (`balancing_energy_in`/`balancing_energy_out`), converted from Wh to W by the timestep,
#  rather than carried as a placeholder here.
RTC_DEFAULT_LIMITS = {
    'balancing_power': 10_000_000_000,  # upper bound on the market deviation variables
    'market_power': None,               # bound on the market power variable (None -> unbounded)
    'hp_power_heat': None,              # heat output fallback when sizing data is missing
    'hp_power_electricity': None,       # electrical input fallback when the COP column is missing
}

# Slack variables on the balance equations let a controller shed or dump energy at a heavy
# penalty rather than fail outright, so a single infeasible agent cannot abort a whole run.
# On by default. Set to false per agent to reproduce runs made before they existed:
#   controller:
#     rtc:
#       slack: false
K_SLACK = 'slack'
DEFAULT_SLACK_ENABLED = True

# Penalty on those slack variables.
# The feedback controller's objective is monetary, so its penalty is a value of lost load in the
# same units as the price forecasts. Prices are integers in 0.01 ct/kWh -- the shipped retailer
# data uses 1992 for a 19.92 ct/kWh retail price -- i.e. 0.1 EUR/MWh per unit. 100_000 is
# therefore 10,000 EUR/MWh, within the usual VOLL range and far above any wholesale price, so
# shedding load is always dearer than serving it.
# The real-time controller's objective is a weighted sum of set-point deviations rather than a
# cost, so its penalty is a dimensionless priority weight. It sits above the largest deviation
# weight (market, 4) so slack is always the least preferred option.
FBC_DEFAULT_SLACK_PENALTY = 100_000
RTC_DEFAULT_SLACK_PENALTY = 10


def resolve_limits(ems_config: dict) -> dict:
    """Bounds for the optimisation variables, from a controller's configuration.

    Unknown keys are kept, so a bound added to a component before it is added to the defaults
    table still reaches it.
    """
    return {**RTC_DEFAULT_LIMITS, **((ems_config or {}).get(K_LIMITS) or {})}


def resolve_slack(ems_config: dict, default_penalty: int) -> tuple:
    """Whether the balance slacks are enabled, and what they cost, for a controller."""
    ems_config = ems_config or {}
    enabled = ems_config.get(K_SLACK, DEFAULT_SLACK_ENABLED)
    penalty = (ems_config.get(K_PENALTIES) or {}).get('slack', default_penalty)

    return bool(enabled), penalty
