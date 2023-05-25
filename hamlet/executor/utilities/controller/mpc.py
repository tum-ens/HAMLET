from linopy import LinearProgram, Variable, Constraint, Objective
from jump import Model, Var, Constraint, solve, maximize
import pyomo.environ as pyo

class Mpc:
    def __init__(self, language: str = None, **kwargs):

        self.options = ['pyomo', 'jump', 'linopy']

        self.language = language if language in self.options else 'jump'

    def solve(self):

        match self.language:
            case 'pyomo':
                MpcPyomo()
            case 'jump':
                MpcJump()
            case 'linopy':
                MpcLinopy()
            case _:
                pass


    ### old functions and stuff from ChatGPT

    def _controller_mpc(self, **kwargs):

        # Obtain data
        data = kwargs['data']
        timetable = kwargs['timetable']
        market = kwargs['market']

        data = self.controller_model_predictive(data=data, timetable=timetable, market=market)

        return data

    def _controller_mpc_rtc(self, **kwargs):

        # Obtain data
        data = kwargs['data']
        timetable = kwargs['timetable']
        market = kwargs['market']

        data = self._controller_rtc(data=data, timetable=timetable, market=market)

        data = self._controller_mpc(data=data, timetable=timetable, market=market)

        return data

    def controller_real_time(self, data, controller="mpc"):
        """Calculate the behaviour of the instance in the previous time step by applying a selected
        controller_real_time strategy to measurement data and plant specifications. Output is supplied in the form of a
        .json file saved to the user's folder, so that execution can be parallelized.

        :return: None
        """

        # default controller_real_time
        if controller == "mpc":
            # grid power is merely the sum of pv and the fixedgen consumers. The battery remains unused.
            df_setpoints = data['setpoints']
            df_setpoints.set_index("timestamp", inplace=True)
            # non-default controllers common model parameters initialized here.
            model = pyo.ConcreteModel()

            # Declare decision variables

            # deviation from setpoint (grid fee-in), absolute components
            model.deviation_gr_plus = pyo.Var(domain=pyo.NonNegativeReals)
            model.deviation_gr_minus = pyo.Var(domain=pyo.NonNegativeReals)

            # pv variables
            model.p_pv = pyo.Var(self._get_list_plants(plant_type="pv"),
                                 domain=pyo.NonNegativeReals)

            # fixedgen decision variables
            model.p_fixedgen = pyo.Var(self._get_list_plants(plant_type="fixedgen"),
                                       domain=pyo.NonNegativeReals)

            # pv maximum power constraint
            def pv_rule(_model, _plant):
                p_max = ft.read_dataframe(f"{self.path}/raw_data_{_plant}.ft")
                p_max.set_index("timestamp", inplace=True)
                p_max = float(p_max[p_max.index == self.ts_delivery_prev]["power"].values)
                p_max *= self.plant_dict[_plant]["power"]
                if self.plant_dict[_plant].get("controllable"):
                    return _model.p_pv[_plant] <= p_max
                return _model.p_pv[_plant] == p_max

            def fixedgen_rule(_model, _plant):
                p_max = ft.read_dataframe(f"{self.path}/raw_data_{_plant}.ft")
                p_max.set_index("timestamp", inplace=True)
                p_max = float(p_max[p_max.index == self.ts_delivery_prev]["power"].values)
                p_max *= self.plant_dict[_plant]["power"]
                if self.plant_dict[_plant].get("controllable"):
                    return _model.p_fixedgen[_plant] <= p_max
                return _model.p_fixedgen[_plant] == p_max

            if self._get_list_plants(plant_type="pv"):
                model.con_pv = pyo.Constraint(self._get_list_plants(plant_type="pv"),
                                              rule=pv_rule)

            if self._get_list_plants(plant_type="fixedgen"):
                model.con_fixedgen = pyo.Constraint(self._get_list_plants(plant_type="fixedgen"),
                                                    rule=fixedgen_rule)

            if self._get_list_plants(plant_type="bat"):
                # battery decision variables
                model.p_bat_in = pyo.Var(self._get_list_plants(plant_type="bat"), domain=pyo.NonNegativeReals)
                model.p_bat_out = pyo.Var(self._get_list_plants(plant_type="bat"), domain=pyo.NonNegativeReals)
                model.p_bat_milp = pyo.Var(self._get_list_plants(plant_type="bat"), domain=pyo.Binary)
                model.deviation_bat_plus = pyo.Var(self._get_list_plants(plant_type="bat"), domain=pyo.NonNegativeReals)
                model.deviation_bat_minus = pyo.Var(self._get_list_plants(plant_type="bat"),
                                                    domain=pyo.NonNegativeReals)

                # else set battery power to zero
                dict_soc_old = {}
                model.n_bat = {}
                model.con_bat_dev = pyo.ConstraintList()

                for bat in self._get_list_plants(plant_type="bat"):
                    with open(f"{self.path}/soc_{bat}.json", "r") as read_file:
                        dict_soc_old[bat] = json.load(read_file)
                    model.n_bat[bat] = self.plant_dict[bat]["efficiency"]

                    model.p_bat_in[bat].setub(self.plant_dict[bat]["power"])
                    model.p_bat_out[bat].setub(self.plant_dict[bat]["power"])
                    model.con_bat_dev.add(expr=model.p_bat_out[bat] - model.p_bat_in[bat]
                                               == df_setpoints[f"power_{bat}"]
                                               - (model.deviation_bat_plus[bat] - model.deviation_bat_minus[bat]))

                def bat_soc_rule_1(_model, _bat):
                    return (dict_soc_old[_bat] - 0.25 * _model.p_bat_out[_bat] / _model.n_bat[_bat]
                            + 0.25 * _model.p_bat_in[_bat] * _model.n_bat[_bat] <= self.plant_dict[_bat].get(
                                "capacity"))

                def bat_soc_rule_2(_model, _bat):
                    return (dict_soc_old[_bat] - 0.25 * _model.p_bat_out[_bat] / _model.n_bat[_bat]
                            + 0.25 * _model.p_bat_in[_bat] * _model.n_bat[_bat] >= 0)

                def bat_bin_rule_minus(_model, _bat):
                    return _model.p_bat_in[_bat] <= 100000 * (1 - _model.p_bat_milp[_bat])

                def bat_bin_rule_plus(_model, _bat):
                    return _model.p_bat_out[_bat] <= 100000 * _model.p_bat_milp[_bat]

                model.bat_soc_1 = pyo.Constraint(self._get_list_plants(plant_type="bat"), rule=bat_soc_rule_1)
                model.bat_soc_2 = pyo.Constraint(self._get_list_plants(plant_type="bat"), rule=bat_soc_rule_2)
                model.bat_bin_minus = pyo.Constraint(self._get_list_plants(plant_type="bat"), rule=bat_bin_rule_minus)
                model.bat_bin_plus = pyo.Constraint(self._get_list_plants(plant_type="bat"), rule=bat_bin_rule_plus)

                # limit battery charging to pv generation
                model.con_batt_charge_grid = pyo.ConstraintList()
                make_const = 0
                expr_left = 0
                expr_right = 0
                for bat in self._get_list_plants(plant_type="bat"):
                    if not self.plant_dict[bat].get("charge_from_grid"):
                        expr_left += model.p_bat_in[bat]
                        make_const = 1
                for pv in self._get_list_plants(plant_type="pv"):
                    expr_right += model.p_pv[pv]
                if make_const:
                    model.con_batt_charge_grid.add(expr=(expr_left <= expr_right))

            # fixedgen load consumption, sum of household loads
            p_load = float(0)
            for hh in self._get_list_plants(plant_type="inflexible_load"):
                p_meas = ft.read_dataframe(f"{self.path}/raw_data_{hh}.ft")
                p_meas.set_index("timestamp", inplace=True)
                p_meas = float(p_meas[p_meas.index == self.ts_delivery_prev]["power"].values)
                p_load += float(p_meas)

            # ev decision variables
            if self._get_list_plants(plant_type="ev"):
                model.p_ev_in = pyo.Var(self._get_list_plants(plant_type="ev"), domain=pyo.NonNegativeReals)
                model.p_ev_out = pyo.Var(self._get_list_plants(plant_type="ev"), domain=pyo.NonNegativeReals)
                model.p_ev_milp = pyo.Var(self._get_list_plants(plant_type="ev"), domain=pyo.Binary)
                model.deviation_ev_plus = pyo.Var(self._get_list_plants(plant_type="ev"), domain=pyo.NonNegativeReals)
                model.deviation_ev_minus = pyo.Var(self._get_list_plants(plant_type="ev"), domain=pyo.NonNegativeReals)
                model.dev_ev_milp = pyo.Var(self._get_list_plants(plant_type="ev"), domain=pyo.Binary)

                model.con_ev_soc = pyo.ConstraintList()
                model.con_ev_milp = pyo.ConstraintList()
                model.con_ev_dev = pyo.ConstraintList()

                model.ev_soc_old = {}
                for ev in self._get_list_plants(plant_type="ev"):
                    model.p_ev_out[ev].setub(0)
                    model.p_ev_in[ev].setub(0)
                    raw_data_ev = ft.read_dataframe(f"{self.path}/raw_data_{ev}.ft")
                    raw_data_ev.set_index("timestamp", inplace=True)
                    raw_data_ev = raw_data_ev[raw_data_ev.index == self.ts_delivery_prev]
                    raw_data_ev = dict(raw_data_ev.loc[self.ts_delivery_prev])
                    if raw_data_ev["availability"] == 1:
                        model.p_ev_in[ev].setub(self.plant_dict[ev]["charging_power"])
                        if self.plant_dict[ev].get("v2g"):
                            model.p_ev_out[ev].setub(self.plant_dict[ev]["charging_power"])

                    with open(f"{self.path}/soc_{ev}.json", "r") as read_file:
                        model.ev_soc_old[ev] = max(0.05 * self.plant_dict[ev]["capacity"],
                                                   json.load(read_file) - raw_data_ev["distance_driven"] / 100
                                                   * self.plant_dict[ev]["consumption"])

                    n_ev = self.plant_dict[ev]["efficiency"]

                    if raw_data_ev["availability"] == 1:
                        model.p_ev_in[ev].setub(self.plant_dict[ev]["charging_power"])
                        if self.plant_dict[ev].get("v2g"):
                            model.p_ev_out[ev].setub(self.plant_dict[ev]["charging_power"])

                        model.con_ev_soc.add(expr=model.ev_soc_old[ev]
                                                  - 0.25 * model.p_ev_out[ev] / n_ev
                                                  + 0.25 * model.p_ev_in[ev] * n_ev
                                                  <= self.plant_dict[ev].get("capacity"))

                        model.con_ev_soc.add(expr=model.ev_soc_old[ev]
                                                  - 0.25 * model.p_ev_out[ev] / n_ev
                                                  + 0.25 * model.p_ev_in[ev] * n_ev
                                                  >= df_setpoints[f"soc_min_{ev}"])

                        model.con_ev_milp.add(expr=model.p_ev_out[ev]
                                                   <= 1000000 * model.p_ev_milp[ev])

                        model.con_ev_milp.add(expr=model.p_ev_in[ev]
                                                   <= 1000000 * (1 - model.p_ev_milp[ev]))

                        model.con_ev_milp.add(expr=model.deviation_ev_plus[ev]
                                                   <= 1000000 * model.dev_ev_milp[ev])

                        model.con_ev_milp.add(expr=model.deviation_ev_minus[ev]
                                                   <= 1000000 * (1 - model.dev_ev_milp[ev]))

                        model.con_ev_dev.add(expr=model.p_ev_out[ev] - model.p_ev_in[ev]
                                                  == df_setpoints[f"power_{ev}"]
                                                  - (model.deviation_ev_plus[ev]
                                                     - model.deviation_ev_minus[ev]))

            # declare balancing constraint, same for all controllers
            def balance_rule(_model):
                expression_left = 0
                if len(self._get_list_plants(plant_type="inflexible_load")):
                    self.meas_val[hh] = int(p_load)
                for _hh in self._get_list_plants(plant_type="inflexible_load"):
                    if self.plant_dict[_hh]["fcast"] != "aggregator":
                        expression_left += p_load
                for _fixedgen in self._get_list_plants(plant_type="fixedgen"):
                    expression_left += _model.p_fixedgen[_fixedgen]
                for _pv in self._get_list_plants(plant_type="pv"):
                    expression_left += model.p_pv[_pv]
                for _bat in self._get_list_plants(plant_type="bat"):
                    expression_left += _model.p_bat_out[_bat] - _model.p_bat_in[_bat]
                for _ev in self._get_list_plants(plant_type="ev"):
                    expression_left += model.p_ev_out[_ev] - model.p_ev_in[_ev]
                expression_right = float(df_setpoints[f"power_{self.config_dict['id_meter_grid']}"])
                expression_right -= _model.deviation_gr_plus - _model.deviation_gr_minus
                return expression_left == expression_right

            model.con_balance = pyo.Constraint(rule=balance_rule)

            # declare objective function, same for all controllers
            # _                             component 1: minimize deviation from target power (0 for self-consumption)
            # _                                          mutual exclusion of absolute components

            def obj_rule(_model):
                obj = 0.5 * (model.deviation_gr_plus + model.deviation_gr_minus)
                for _bat in self._get_list_plants(plant_type="bat"):
                    obj += 0.1 * _model.deviation_bat_minus[_bat]
                for _ev in self._get_list_plants(plant_type="ev"):
                    obj += _model.deviation_ev_minus[_ev]
                return obj

            model.objective_fun = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

            # solve model
            pyo.SolverFactory(self.config_dict["solver"]).solve(model)

            # assign results to instance variables for logging
            meas_grid = p_load
            for pv in self._get_list_plants(plant_type="pv"):
                self.meas_val[pv] = model.p_pv[pv].value
                meas_grid += model.p_pv[pv].value
            for fixedgen in self._get_list_plants(plant_type="fixedgen"):
                self.meas_val[fixedgen] = model.p_fixedgen[fixedgen].value
                meas_grid += model.p_fixedgen[fixedgen].value
            for bat in self._get_list_plants(plant_type="bat"):
                self.meas_val[bat] = model.p_bat_out[bat].value - model.p_bat_in[bat].value
                meas_grid += model.p_bat_out[bat].value - model.p_bat_in[bat].value
                bat_soc_new = dict_soc_old[bat] \
                              - 0.25 * model.p_bat_out[bat].value / self.plant_dict[bat]["efficiency"] \
                              + 0.25 * model.p_bat_in[bat].value * self.plant_dict[bat]["efficiency"]
                with open(f"{self.path}/soc_{bat}.json", "w") as write_file:
                    json.dump(bat_soc_new, write_file)
            for ev in self._get_list_plants(plant_type="ev"):
                self.meas_val[ev] = model.p_ev_out[ev].value - model.p_ev_in[ev].value
                meas_grid += model.p_ev_out[ev].value - model.p_ev_in[ev].value
                ev_soc_new = model.ev_soc_old[ev] \
                             - 0.25 * model.p_ev_out[ev].value / self.plant_dict[ev]["efficiency"] \
                             + 0.25 * model.p_ev_in[ev].value * self.plant_dict[ev]["efficiency"]
                with open(f"{self.path}/soc_{ev}.json", "w") as write_file:
                    json.dump(ev_soc_new, write_file)

            self.meas_val[self.config_dict['id_meter_grid']] = int(meas_grid)

        else:
            # fixedgen load consumption, sum of household loads
            p_load = float(0)
            for hh in self._get_list_plants(plant_type="inflexible_load"):
                p_meas = ft.read_dataframe(f"{self.path}/raw_data_{hh}.ft")
                p_meas.set_index("timestamp", inplace=True)
                p_meas = float(p_meas[p_meas.index == self.ts_delivery_prev]["power"].values)
                self.meas_val[hh] = p_meas
                p_load += p_meas

            self.meas_val[self.config_dict['id_meter_grid']] = int(p_load)

        # save calculated values to .json file so that results can be used by later methods in case of parallelization
        with open(f"{self.path}/controller_rtc.json", "w") as write_file:
            json.dump(self.meas_val, write_file)

    def controller_model_predictive(self, data, controller="mpc"):
        """Execute the model predictive controller_real_time for the market participant given the predicted
        generation, consumption, and market prices for a configurable time horizon.

        The controller_real_time will always attempt to maximize its earnings. If no market price optimization is
        desired, a flat market price forecasting should be input.

        :return: None
        """
        if controller == "mpc":
            # Declare the pyomo model
            model = pyo.ConcreteModel()
            # declare decision variables (vectors of same length as MPC horizon)
            # pv power variable

            if self._get_list_plants(plant_type="pv"):
                model.p_pv = pyo.Var(self._get_list_plants(plant_type="pv"),
                                     range(0, self.config_dict["mpc_horizon"]),
                                     domain=pyo.NonNegativeReals)

            # fixedgen power variable
            if self._get_list_plants(plant_type="fixedgen"):
                model.p_fixedgen = pyo.Var(self._get_list_plants(plant_type="fixedgen"),
                                           range(0, self.config_dict["mpc_horizon"]),
                                           domain=pyo.NonNegativeReals)

            # battery power, absolute components
            if self._get_list_plants(plant_type="bat"):
                model.p_bat_in = pyo.Var(self._get_list_plants(plant_type="bat"),
                                         range(self.config_dict["mpc_horizon"]),
                                         domain=pyo.NonNegativeReals)
                model.p_bat_out = pyo.Var(self._get_list_plants(plant_type="bat"),
                                          range(self.config_dict["mpc_horizon"]),
                                          domain=pyo.NonNegativeReals)
                model.p_bat_milp = pyo.Var(self._get_list_plants(plant_type="bat"),
                                           range(self.config_dict["mpc_horizon"]),
                                           domain=pyo.Binary)
                model.soc_bat = pyo.Var(self._get_list_plants(plant_type="bat"),
                                        range(self.config_dict["mpc_horizon"]),
                                        domain=pyo.NonNegativeReals)

            for bat in self._get_list_plants(plant_type="bat"):
                for i in range(self.config_dict["mpc_horizon"]):
                    model.p_bat_in[bat, i].setub(self.plant_dict[bat]["power"])
                    model.p_bat_out[bat, i].setub(self.plant_dict[bat]["power"])
                    model.soc_bat[bat, i].setub(self.plant_dict[bat]["capacity"])

            # EV decision variables, absolute components
            if self._get_list_plants(plant_type="ev"):
                model.p_ev_in = pyo.Var(self._get_list_plants(plant_type="ev"),
                                        range(self.config_dict["mpc_horizon"]),
                                        domain=pyo.NonNegativeReals)
                model.p_ev_out = pyo.Var(self._get_list_plants(plant_type="ev"),
                                         range(self.config_dict["mpc_horizon"]),
                                         domain=pyo.NonNegativeReals)
                model.p_ev_milp = pyo.Var(self._get_list_plants(plant_type="ev"),
                                          range(self.config_dict["mpc_horizon"]),
                                          domain=pyo.Binary)
                model.soc_ev = pyo.Var(self._get_list_plants(plant_type="ev"),
                                       range(self.config_dict["mpc_horizon"]),
                                       domain=pyo.NonNegativeReals)
                model.ev_slack = pyo.Var(self._get_list_plants(plant_type="ev"),
                                         range(self.config_dict["mpc_horizon"]),
                                         domain=pyo.NonNegativeReals)
                model.con_soc_ev = pyo.ConstraintList()

                model.con_p_ev_minus = pyo.ConstraintList()
                model.con_p_ev_plus = pyo.ConstraintList()

            dict_soc_ev_min = {}
            dict_soc_ev_old = {}
            for ev in self._get_list_plants(plant_type="ev"):
                with open(f"{self.path}/soc_{ev}.json", "r") as read_file:
                    dict_soc_ev_old[ev] = json.load(read_file)

                n_ev = self.plant_dict[ev]["efficiency"]
                dict_soc_ev_min[ev] = [0] * self.config_dict["mpc_horizon"]
                soc_potential = dict_soc_ev_old[ev]

                for i in range(self.config_dict["mpc_horizon"]):
                    # Compute the maximum possible distance that the EV can drive to return with a minimum SoC of 5 %. This
                    #   simulates that prosumers charge their car outside of home when driving longer distances
                    max_distance = (soc_potential - 0.05 * self.plant_dict[ev]["capacity"]) \
                                   / self.plant_dict[ev].get("consumption") * 100

                    # Save value to new column that contains the adjusted distances
                    self.mpc_table.loc[self.mpc_table.index[i], f"distance_driven_adjusted_{ev}"] = min(
                        self.mpc_table[f"distance_driven_{ev}"].iloc[i], max_distance)

                    # Add constraint that SoC can never surpass the maximum capacity of the vehicle
                    model.con_soc_ev.add(expr=model.soc_ev[ev, i] <= self.plant_dict[ev].get("capacity"))

                    # Subtract consumption based on the driven kilometers since last time step (0.25 --> 15 min --> 900 s)
                    soc_potential -= self.plant_dict[ev].get("consumption") \
                        * self.mpc_table[f"distance_driven_adjusted_{ev}"].iloc[i] * 1 / 100

                    # Add charged energy since last time step (0.25 --> 15 min --> 900 s)
                    soc_potential += self.plant_dict[ev].get("charging_power") * 0.25 * n_ev \
                        * self.mpc_table[f"availability_{ev}"].iloc[i]

                    # Keep charged energy between 5 % and 85 % of maximum capacity
                    soc_potential = max(min(soc_potential, 0.85 * self.plant_dict[ev].get("capacity")),
                                        0.05 * self.plant_dict[ev].get("capacity"))

                    # Case 1: currently not last time step, EV is available but will have left next time step
                    # Case 2: currently last time step, EV is available
                    # Action: Add constraint that SoC needs to be at least the potential SoC
                    if (i < self.config_dict["mpc_horizon"] - 1 and self.mpc_table[f"availability_{ev}"].iloc[i] == 1
                        and self.mpc_table[f"availability_{ev}"].iloc[i + 1] == 0) \
                       or (i == self.config_dict["mpc_horizon"] - 1 and self.mpc_table[f"availability_{ev}"].iloc[i] == 1):
                        model.con_soc_ev.add(expr=model.soc_ev[ev, i] >= soc_potential - model.ev_slack[ev, i])
                        dict_soc_ev_min[ev][i] = soc_potential

                    # Set availability and powers depending on if EV is available or not
                    if self.mpc_table[f"availability_{ev}"].iloc[i] == 1:
                        # Set if EV can charge or discharge
                        model.con_p_ev_minus.add(expr=model.p_ev_in[ev, i] <= 1000000 * (1 - model.p_ev_milp[ev, i]))
                        model.con_p_ev_plus.add(expr=model.p_ev_out[ev, i] <= 1000000 * model.p_ev_milp[ev, i])

                        # Set maximum charging and discharging power (upper boundaries)
                        model.p_ev_in[ev, i].setub(self.plant_dict[ev].get("charging_power"))
                        if self.plant_dict[ev].get("v2g"):
                            model.p_ev_out[ev, i].setub(self.plant_dict[ev].get("charging_power"))
                        else:
                            model.p_ev_out[ev, i].setub(0)
                    else:
                        model.p_ev_in[ev, i] = float(0)
                        model.p_ev_out[ev, i] = float(0)

                # Loop from end to beginning and increase the min SoC for the previous timestep to ensure that the EV starts
                #   charging soon enough to reach the final min SoC before the car leaves
                for i in range(self.config_dict["mpc_horizon"]-1, 0, -1):
                    if dict_soc_ev_min[ev][i] > 0:
                        dict_soc_ev_min[ev][i - 1] = dict_soc_ev_min[ev][i] - \
                                                     0.25 * self.plant_dict[ev]["charging_power"] * n_ev
                        dict_soc_ev_min[ev][i - 1] = max(dict_soc_ev_min[ev][i - 1], 0)

            # Add constraint that the SoC of each time step needs to be the SoC of the previous one plus the charge and
            #   minus the discharge and the consumption due to the driven distance
            model.con_soc_ev_calc = pyo.ConstraintList()
            for ev in self._get_list_plants(plant_type="ev"):
                n_ev = self.plant_dict[ev]["efficiency"]
                model.con_soc_ev_calc.add(expr=dict_soc_ev_old[ev]
                                          - 0.25 * model.p_ev_out[ev, 0] / n_ev
                                          + 0.25 * model.p_ev_in[ev, 0] * n_ev
                                          - (self.plant_dict[ev].get("consumption")
                                             * self.mpc_table[f"distance_driven_adjusted_{ev}"].iloc[0]
                                             * 1 / 100)
                                          == model.soc_ev[ev, 0])
                for t in range(1, self.config_dict["mpc_horizon"]):
                    model.con_soc_ev_calc.add(expr=model.soc_ev[ev, t - 1]
                                              - 0.25 * model.p_ev_out[ev, t] / n_ev
                                              + 0.25 * model.p_ev_in[ev, t] * n_ev
                                              - (self.plant_dict[ev].get("consumption")
                                                 * self.mpc_table[f"distance_driven_adjusted_{ev}"].iloc[t]
                                                 * 1 / 100)
                                              == model.soc_ev[ev, t])

            # Add variables for grid powerflow
            model.p_grid_out = pyo.Var(range(self.config_dict["mpc_horizon"]), domain=pyo.NonNegativeReals)
            model.p_grid_in = pyo.Var(range(self.config_dict["mpc_horizon"]), domain=pyo.NonNegativeReals)
            model.p_grid_milp = pyo.Var(range(self.config_dict["mpc_horizon"]), domain=pyo.Binary)

            # fixedgen power, sum of household loads
            p_load = [0] * self.config_dict["mpc_horizon"]
            for hh in self._get_list_plants(plant_type="inflexible_load"):
                for i, ts_d in enumerate(range(self.ts_delivery_current,
                                               self.ts_delivery_current + self.config_dict["mpc_horizon"]*900, 900)):
                    p_load[i] += float(self.mpc_table.loc[ts_d, f"power_{hh}"])

            model.con_p_bat_bin = pyo.ConstraintList()
            for bat in self._get_list_plants(plant_type="bat"):
                for t in range(self.config_dict["mpc_horizon"]):
                    model.con_p_bat_bin.add(expr=model.p_bat_in[bat, t] <= 1000000 * (1 - model.p_bat_milp[bat, t]))
                    model.con_p_bat_bin.add(expr=model.p_bat_out[bat, t] <= 1000000 * model.p_bat_milp[bat, t])

            # Declare model constraints
            model.con_grid_bin = pyo.ConstraintList()
            for t in range(self.config_dict["mpc_horizon"]):
                model.con_grid_bin.add(expr=model.p_grid_in[t] <= 1000000 * (1 - model.p_grid_milp[t]))
                model.con_grid_bin.add(expr=model.p_grid_out[t] <= 1000000 * model.p_grid_milp[t])

            # define pv power upper bound from input file
            model.con_p_pv = pyo.ConstraintList()
            model.sum_pv = [0] * self.config_dict["mpc_horizon"]
            for pv in self._get_list_plants(plant_type="pv"):
                for t, t_d in enumerate(range(self.ts_delivery_current,
                                              self.ts_delivery_current + 900*self.config_dict["mpc_horizon"], 900)):
                    model.sum_pv[t] += self.mpc_table.loc[t_d, f"power_{pv}"]
                    if self.plant_dict[pv].get("controllable"):
                        model.con_p_pv.add(expr=model.p_pv[pv, t] <= round(self.mpc_table.loc[t_d, f"power_{pv}"], 1))
                    else:
                        model.con_p_pv.add(expr=model.p_pv[pv, t] == round(self.mpc_table.loc[t_d, f"power_{pv}"], 1))

            # define fixedgen power upper bound from input file
            model.con_p_fixedgen = pyo.ConstraintList()
            model.sum_fixedgen = [0] * self.config_dict["mpc_horizon"]
            for fixedgen in self._get_list_plants(plant_type="fixedgen"):
                for t, t_d in enumerate(range(self.ts_delivery_current,
                                              self.ts_delivery_current + 900*self.config_dict["mpc_horizon"], 900)):
                    model.sum_fixedgen[t] += self.mpc_table.loc[t_d, f"power_{fixedgen}"]
                    if self.plant_dict[fixedgen].get("controllable"):
                        model.con_p_fixedgen.add(expr=model.p_fixedgen[fixedgen, t]
                                                 <= self.mpc_table.loc[t_d, f"power_{fixedgen}"])
                    else:
                        model.con_p_fixedgen.add(expr=model.p_fixedgen[fixedgen, t]
                                                 == self.mpc_table.loc[t_d, f"power_{fixedgen}"])

            # limit battery charging to pv generation
            model.con_batt_charge_grid = pyo.ConstraintList()
            for t in range(0, self.config_dict["mpc_horizon"]):
                expr_left = 0
                expr_right = model.sum_pv[t]
                make_const = 0
                for bat in self._get_list_plants(plant_type="bat"):
                    if not self.plant_dict[bat].get("charge_from_grid"):
                        expr_left += model.p_bat_in[bat, t]
                        make_const = 1
                if make_const:
                    model.con_batt_charge_grid.add(expr=(expr_left <= expr_right))

            # define initial battery soc, determined using first term of battery power and battery soc in the prev step
            model.con_soc_calc = pyo.ConstraintList()

            for bat in self._get_list_plants(plant_type="bat"):
                n_bat = self.plant_dict[bat]["efficiency"]
                with open(f"{self.path}/soc_{bat}.json", "r") as read_file:
                    soc_bat_init = json.load(read_file)
                model.con_soc_calc.add(expr=soc_bat_init
                                       - 0.25 * model.p_bat_out[bat, 0] / n_bat
                                       + 0.25 * model.p_bat_in[bat, 0] * n_bat
                                       == model.soc_bat[bat, 0])
                for t in range(1, self.config_dict["mpc_horizon"]):
                    model.con_soc_calc.add(expr=model.soc_bat[bat, t - 1]
                                           - 0.25 * model.p_bat_out[bat, t] / n_bat
                                           + 0.25 * model.p_bat_in[bat, t] * n_bat
                                           == model.soc_bat[bat, t])

            model.con_balance = pyo.ConstraintList()
            for _t in range(self.config_dict["mpc_horizon"]):
                expression_left = p_load[_t]
                for _pv in self._get_list_plants(plant_type="pv"):
                    expression_left += model.p_pv[_pv, _t]
                for _bat in self._get_list_plants(plant_type="bat"):
                    expression_left += model.p_bat_out[_bat, _t] - model.p_bat_in[_bat, _t]
                for _ev in self._get_list_plants(plant_type="ev"):
                    expression_left += model.p_ev_out[_ev, _t] - model.p_ev_in[_ev, _t]
                for _fixedgen in self._get_list_plants(plant_type="fixedgen"):
                    expression_left += model.p_fixedgen[_fixedgen, _t]
                expression_right = model.p_grid_out[_t] - model.p_grid_in[_t]
                model.con_balance.add(expr=(expression_left == expression_right))

            model.price = list(self.mpc_table["price"])
            model.price_levies_pos = list(self.mpc_table["price_energy_levies_positive"])
            model.price_levies_neg = list(self.mpc_table["price_energy_levies_negative"])

            # Define objective function
            def obj_rule(_model):
                step_obj = 0
                for j in range(0, self.config_dict["mpc_horizon"]):
                    #            component 1:   grid feed in valued at predicted price
                    #            component 2:   grid consumption valued at predicted price plus fixed levies
                    step_obj += _model.p_grid_out[j] * (-_model.price[j] + model.price_levies_pos[j]) \
                                + _model.p_grid_in[j] * (_model.price[j] + model.price_levies_neg[j])
                    # ensure non-degeneracy of the MILP
                    # cannot be used by GLPK as non-linear objectives cannot be solved
                    if self.config_dict["solver"] != "glpk":
                        step_obj += 0.0005 * _model.p_grid_out[j] * _model.p_grid_out[j] / 1000 / 1000
                        step_obj += 0.0005 * _model.p_grid_in[j] * _model.p_grid_in[j] / 1000 / 1000
                    # legacy check for electric vehicle constraint violation
                    for item in self._get_list_plants(plant_type="ev"):
                        step_obj += _model.ev_slack[item, j] * 100000000
                return step_obj

            # Solve model
            model.objective_fun = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
            pyo.SolverFactory(self.config_dict["solver"]).solve(model)

            # Update mpc_table with results of model
            dict_mpc_table = self.mpc_table.to_dict()
            for i, t_d in enumerate(range(self.ts_delivery_current,
                                          self.ts_delivery_current + 900 * self.config_dict["mpc_horizon"], 900)):
                # PV
                for pv in self._get_list_plants(plant_type="pv"):
                    dict_mpc_table[f"power_{pv}"][t_d] = model.p_pv[pv, i]()

                # Battery
                for bat in self._get_list_plants(plant_type="bat"):
                    dict_mpc_table[f"power_{bat}"][t_d] = model.p_bat_out[bat, i]() - model.p_bat_in[bat, i]()
                    dict_mpc_table[f"soc_{bat}"][t_d] = model.soc_bat[bat, i]()

                # EV
                for ev in self._get_list_plants(plant_type="ev"):
                    dict_mpc_table[f"power_{ev}"][t_d] = model.p_ev_out[ev, i]() - model.p_ev_in[ev, i]()
                    dict_mpc_table[f"soc_{ev}"][t_d] = model.soc_ev[ev, i]()
                    dict_mpc_table[f"soc_min_{ev}"][t_d] = min(dict_soc_ev_min[ev][i], self.plant_dict[ev].get("capacity"))
                    # Temporary check for errors in the electric vehicle charging routine

                    if model.ev_slack[ev, i]() >= 10**-6:
                        # for object name file1.
                        logfile = open(f"{'/'.join(self.path.split('/')[:-2])}/log.txt", "a")
                        logfile.write(f"Warning: User {self.config_dict['id_user']}'s EV #{ev} violated its charging "
                                      f"constraint at {self.t_now} resulting in a slack value of {model.ev_slack[ev, i]()} "
                                      f"in MPC step {i}")
                        logfile.close()
                # Fixed generation
                for fixedgen in self._get_list_plants(plant_type="fixedgen"):
                    dict_mpc_table[f"power_{fixedgen}"][t_d] = model.p_fixedgen[fixedgen, i]()

                # Grid power
                dict_mpc_table[f"power_{self.config_dict['id_meter_grid']}"][t_d] \
                    = model.p_grid_out[i]() - model.p_grid_in[i]()

            # Save results to file, which will be used as basis for controller_real_time set points and market trading
            self.mpc_table = pd.DataFrame.from_dict(dict_mpc_table)
        elif controller == "inflexible_load":
            # fixedgen power, sum of household loads
            p_load = [0] * self.config_dict["mpc_horizon"]
            for hh in self._get_list_plants(plant_type="inflexible_load"):
                for i, ts_d in enumerate(range(self.ts_delivery_current,
                                               self.ts_delivery_current + self.config_dict["mpc_horizon"]*900, 900)):
                    p_load[i] += float(self.mpc_table.loc[ts_d, f"power_{hh}"])
            dict_mpc_table = self.mpc_table.to_dict()
            dict_mpc_table = self.mpc_table.to_dict()
            for i, t_d in enumerate(range(self.ts_delivery_current,
                                          self.ts_delivery_current + 900 * self.config_dict["mpc_horizon"], 900)):
                # Grid power
                dict_mpc_table[f"power_{self.config_dict['id_meter_grid']}"][t_d] = p_load[i]
            # Save results to file, which will be used as basis for controller_real_time set points and market trading
            self.mpc_table = pd.DataFrame.from_dict(dict_mpc_table)

        ft.write_dataframe(self.mpc_table.reset_index().rename(columns={"index": "timestamp"}),
                           f"{self.path}/controller_mpc.ft")

    def controller_model_predictive_jump(self, data, controller="mpc"):
        """
        Execute the model predictive controller_real_time for the market participant given the predicted
        generation, consumption, and market prices for a configurable time horizon.

        The controller_real_time will always attempt to maximize its earnings. If no market price optimization is
        desired, a flat market price forecasting should be input.

        :return: None
        """

        if controller == "mpc":
            # Create the model
            model = Model()

            # Declare decision variables (vectors of the same length as MPC horizon)

            # pv power variable
            if self._get_list_plants(plant_type="pv"):
                p_pv = {}
                for plant in self._get_list_plants(plant_type="pv"):
                    p_pv[plant] = {}
                    for i in range(0, self.config_dict["mpc_horizon"]):
                        p_pv[plant][i] = Variable(domain=NonNegativeReals)

            # fixedgen power variable
            if self._get_list_plants(plant_type="fixedgen"):
                p_fixedgen = {}
                for plant in self._get_list_plants(plant_type="fixedgen"):
                    p_fixedgen[plant] = {}
                    for i in range(0, self.config_dict["mpc_horizon"]):
                        p_fixedgen[plant][i] = Variable(domain=NonNegativeReals)

            # battery power, absolute components
            if self._get_list_plants(plant_type="bat"):
                p_bat_in = {}
                p_bat_out = {}
                p_bat_milp = {}
                soc_bat = {}
                for plant in self._get_list_plants(plant_type="bat"):
                    p_bat_in[plant] = {}
                    p_bat_out[plant] = {}
                    p_bat_milp[plant] = {}
                    soc_bat[plant] = {}
                    for i in range(self.config_dict["mpc_horizon"]):
                        p_bat_in[plant][i] = Variable(domain=NonNegativeReals)
                        p_bat_out[plant][i] = Variable(domain=NonNegativeReals)
                        p_bat_milp[plant][i] = Variable(domain=Binary)
                        soc_bat[plant][i] = Variable(domain=NonNegativeReals)

            for bat in self._get_list_plants(plant_type="bat"):
                for i in range(self.config_dict["mpc_horizon"]):
                    p_bat_in[bat][i].setub(self.plant_dict[bat]["power"])
                    p_bat_out[bat][i].setub(self.plant_dict[bat]["power"])
                    soc_bat[bat][i].setub(self.plant_dict[bat]["capacity"])

            # EV decision variables, absolute components
            if self._get_list_plants(plant_type="ev"):
                p_ev_in = {}
                p_ev_out = {}
                p_ev_milp = {}
                soc_ev = {}
                ev_slack = {}
                con_soc_ev = []
                con_p_ev_minus = []
                con_p_ev_plus = []
                for plant in self._get_list_plants(plant_type="ev"):
                    p_ev_in[plant] = {}
                    p_ev_out[plant] = {}
                    p_ev_milp[plant] = {}
                    soc_ev[plant] = {}
                    ev_slack[plant] = {}
                    con_soc_ev.append(ConstraintList())
                    con_p_ev_minus.append(ConstraintList())
                    con_p_ev_plus.append(ConstraintList())
                    for i in range(self.config_dict["mpc_horizon"]):
                        p_ev_in[plant][i] = Variable(domain=NonNegativeReals)
                        p_ev_out[plant][i] = Variable(domain=NonNegativeReals)
                        p_ev_milp[plant][i] = Variable(domain=Binary)
                        soc_ev[plant][i] = Variable(domain=NonNegativeReals)
                        ev_slack[plant][i] = Variable(domain=NonNegativeReals)

            for ev in self._get_list_plants(plant_type="ev"):
                for i in range(self.config_dict["mpc_horizon"]):
                    p_ev_in[ev][i].setub(self.plant_dict[ev]["power"])
                    p_ev_out[ev][i].setub(self.plant_dict[ev]["power"])
                    soc_ev[ev][i].setub(self.plant_dict[ev]["capacity"])

            # balance constraint
            con_balance = ConstraintList()
            for i in range(self.config_dict["mpc_horizon"]):
                con_balance.append(
                    sum(p_pv[plant][i] for plant in self._get_list_plants(plant_type="pv")) +
                    sum(p_fixedgen[plant][i] for plant in self._get_list_plants(plant_type="fixedgen")) +
                    sum(p_bat_in[plant][i] - p_bat_out[plant][i] for plant in self._get_list_plants(plant_type="bat")) +
                    sum(p_ev_in[plant][i] - p_ev_out[plant][i] for plant in self._get_list_plants(plant_type="ev")) ==
                    data["predicted_load"][i]
                )

            # Constraints for absolute values
            # battery state of charge
            con_soc_bat = []
            for bat in self._get_list_plants(plant_type="bat"):
                con_soc_bat.append(ConstraintList())
                for i in range(self.config_dict["mpc_horizon"]):
                    if i == 0:
                        con_soc_bat[-1].append(
                            soc_bat[bat][i] ==
                            self.plant_dict[bat]["soc_init"] +
                            (p_bat_in[bat][i] - p_bat_out[bat][i]) * self.config_dict["mpc_sampling_time"] /
                            self.plant_dict[bat]["capacity"]
                        )
                    else:
                        con_soc_bat[-1].append(
                            soc_bat[bat][i] ==
                            soc_bat[bat][i - 1] +
                            (p_bat_in[bat][i] - p_bat_out[bat][i]) * self.config_dict["mpc_sampling_time"] /
                            self.plant_dict[bat]["capacity"]
                        )

            # EV state of charge
            con_soc_ev = []
            for ev in self._get_list_plants(plant_type="ev"):
                con_soc_ev.append(ConstraintList())
                for i in range(self.config_dict["mpc_horizon"]):
                    if i == 0:
                        con_soc_ev[-1].append(
                            soc_ev[ev][i] ==
                            self.plant_dict[ev]["soc_init"] +
                            (p_ev_in[ev][i] - p_ev_out[ev][i]) * self.config_dict["mpc_sampling_time"] /
                            self.plant_dict[ev]["capacity"]
                        )
                    else:
                        con_soc_ev[-1].append(
                            soc_ev[ev][i] ==
                            soc_ev[ev][i - 1] +
                            (p_ev_in[ev][i] - p_ev_out[ev][i]) * self.config_dict["mpc_sampling_time"] /
                            self.plant_dict[ev]["capacity"]
                        )

            # Constraints for power limits
            con_power_limit = ConstraintList()

            # PV power limits
            for plant in self._get_list_plants(plant_type="pv"):
                for i in range(self.config_dict["mpc_horizon"]):
                    con_power_limit.append(p_pv[plant][i] <= self.plant_dict[plant]["power"])

            # Fixed generation power limits
            for plant in self._get_list_plants(plant_type="fixedgen"):
                for i in range(self.config_dict["mpc_horizon"]):
                    con_power_limit.append(p_fixedgen[plant][i] <= self.plant_dict[plant]["power"])

            # Battery power limits
            for bat in self._get_list_plants(plant_type="bat"):
                for i in range(self.config_dict["mpc_horizon"]):
                    con_power_limit.append(p_bat_in[bat][i] <= self.plant_dict[bat]["power"])
                    con_power_limit.append(p_bat_out[bat][i] <= self.plant_dict[bat]["power"])

            # EV power limits
            for ev in self._get_list_plants(plant_type="ev"):
                for i in range(self.config_dict["mpc_horizon"]):
                    con_power_limit.append(p_ev_in[ev][i] <= self.plant_dict[ev]["power"])
                    con_power_limit.append(p_ev_out[ev][i] <= self.plant_dict[ev]["power"])

            # Constraints for MILP variables
            con_milp = ConstraintList()

            # Battery MILP constraints
            for bat in self._get_list_plants(plant_type="bat"):
                for i in range(self.config_dict["mpc_horizon"]):
                    con_milp.append(p_bat_in[bat][i] <= p_bat_milp[bat][i] * self.plant_dict[bat]["power"])
                    con_milp.append(p_bat_out[bat][i] <= (1 - p_bat_milp[bat][i]) * self.plant_dict[bat]["power"])

            # EV MILP constraints
            for ev in self._get_list_plants(plant_type="ev"):
                for i in range(self.config_dict["mpc_horizon"]):
                    con_milp.append(p_ev_in[ev][i] <= p_ev_milp[ev][i] * self.plant_dict[ev]["power"])
                    con_milp.append(p_ev_out[ev][i] <= (1 - p_ev_milp[ev][i]) * self.plant_dict[ev]["power"])

            # Objective function
            obj = Objective(
                sum(
                    sum(
                        (p_pv[plant][i] - data["predicted_pv"][plant][i]) *
                        (self.config_dict["mpc_price"]["pv_buy"] - self.config_dict["mpc_price"]["pv_sell"])
                        for plant in self._get_list_plants(plant_type="pv")
                    )
                    +
                    sum(
                        (p_fixedgen[plant][i] - data["predicted_fixedgen"][plant][i]) *
                        (self.config_dict["mpc_price"]["fixedgen_buy"] - self.config_dict["mpc_price"]["fixedgen_sell"])
                        for plant in self._get_list_plants(plant_type="fixedgen")
                    )
                    +
                    sum(
                        (p_bat_in[plant][i] - p_bat_out[plant][i] - data["predicted_bat"][plant][i]) *
                        (self.config_dict["mpc_price"]["bat_buy"] - self.config_dict["mpc_price"]["bat_sell"])
                        for plant in self._get_list_plants(plant_type="bat")
                    )
                    +
                    sum(
                        (p_ev_in[plant][i] - p_ev_out[plant][i] - data["predicted_ev"][plant][i]) *
                        (self.config_dict["mpc_price"]["ev_buy"] - self.config_dict["mpc_price"]["ev_sell"])
                        for plant in self._get_list_plants(plant_type="ev")
                    )
                    for i in range(self.config_dict["mpc_horizon"])
                ),
                sense=maximize
            )

            # Add variables, constraints, and objective to the model
            for plant in self._get_list_plants(plant_type="pv"):
                for i in range(self.config_dict["mpc_horizon"]):
                    model.add(p_pv[plant][i])
            for plant in self._get_list_plants(plant_type="fixedgen"):
                for i in range(self.config_dict["mpc_horizon"]):
                    model.add(p_fixedgen[plant][i])
            for plant in self._get_list_plants(plant_type="bat"):
                for i in range(self.config_dict["mpc_horizon"]):
                    model.add(p_bat_in[plant][i])
                    model.add(p_bat_out[plant][i])
                    model.add(p_bat_milp[plant][i])
                    model.add(soc_bat[plant][i])
            for plant in self._get_list_plants(plant_type="ev"):
                for i in range(self.config_dict["mpc_horizon"]):
                    model.add(p_ev_in[plant][i])
                    model.add(p_ev_out[plant][i])
                    model.add(p_ev_milp[plant][i])
                    model.add(soc_ev[plant][i])
                    model.add(ev_slack[plant][i])
            model.add(con_balance)
            for con in con_soc_bat:
                model.add(con)
            for con in con_soc_ev:
                model.add(con)
            model.add(con_power_limit)
            model.add(con_milp)
            model.add(obj)

            # Solve the model
            model.optimize()

            # Retrieve and store the optimized values
            self.controller_values["p_pv"] = {}
            for plant in self._get_list_plants(plant_type="pv"):
                self.controller_values["p_pv"][plant] = [
                    p_pv[plant][i].value for i in range(self.config_dict["mpc_horizon"])
                ]

            self.controller_values["p_fixedgen"] = {}
            for plant in self._get_list_plants(plant_type="fixedgen"):
                self.controller_values["p_fixedgen"][plant] = [
                    p_fixedgen[plant][i].value for i in range(self.config_dict["mpc_horizon"])
                ]

            self.controller_values["p_bat_in"] = {}
            for plant in self._get_list_plants(plant_type="bat"):
                self.controller_values["p_bat_in"][plant] = [
                    p_bat_in[plant][i].value for i in range(self.config_dict["mpc_horizon"])
                ]

            self.controller_values["p_bat_out"] = {}
            for plant in self._get_list_plants(plant_type="bat"):
                self.controller_values["p_bat_out"][plant] = [
                    p_bat_out[plant][i].value for i in range(self.config_dict["mpc_horizon"])
                ]

            self.controller_values["p_bat_milp"] = {}
            for plant in self._get_list_plants(plant_type="bat"):
                self.controller_values["p_bat_milp"][plant] = [
                    p_bat_milp[plant][i].value for i in range(self.config_dict["mpc_horizon"])
                ]

            self.controller_values["soc_bat"] = {}
            for plant in self._get_list_plants(plant_type="bat"):
                self.controller_values["soc_bat"][plant] = [
                    soc_bat[plant][i].value for i in range(self.config_dict["mpc_horizon"])
                ]

            self.controller_values["p_ev_in"] = {}
            for plant in self._get_list_plants(plant_type="ev"):
                self.controller_values["p_ev_in"][plant] = [
                    p_ev_in[plant][i].value for i in range(self.config_dict["mpc_horizon"])
                ]

            self.controller_values["p_ev_out"] = {}
            for plant in self._get_list_plants(plant_type="ev"):
                self.controller_values["p_ev_out"][plant] = [
                    p_ev_out[plant][i].value for i in range(self.config_dict["mpc_horizon"])
                ]

            self.controller_values["p_ev_milp"] = {}
            for plant in self._get_list_plants(plant_type="ev"):
                self.controller_values["p_ev_milp"][plant] = [
                    p_ev_milp[plant][i].value for i in range(self.config_dict["mpc_horizon"])
                ]

            self.controller_values["soc_ev"] = {}
            for plant in self._get_list_plants(plant_type="ev"):
                self.controller_values["soc_ev"][plant] = [
                    soc_ev[plant][i].value for i in range(self.config_dict["mpc_horizon"])
                ]

            self.controller_values["ev_slack"] = {}
            for plant in self._get_list_plants(plant_type="ev"):
                self.controller_values["ev_slack"][plant] = [
                    ev_slack[plant][i].value for i in range(self.config_dict["mpc_horizon"])
                ]

        elif controller == "other":
            # Implementation of another type of controller
            pass

        else:
            raise ValueError("Invalid controller type specified.")


class MpcPyomo:
    pass

class MpcJump:  # The Jump implementation with subfunctions
    def __init__(self):
        self.controller_values = {}

    def controller_model_predictive(self, controller):
        if controller == "mpc":
            self._mpc_controller()

        elif controller == "other":
            # Implementation of another type of controller
            pass

        else:
            raise ValueError("Invalid controller type specified.")

    def _mpc_controller(self):
        # Define and solve the model predictive controller using MPC
        model = Model()

        # Decision Variables
        p_pv = self._define_pv_power_vars()
        p_fixedgen = self._define_fixedgen_power_vars()
        p_bat_in, p_bat_out, p_bat_milp, soc_bat = self._define_battery_vars()
        p_ev_in, p_ev_out, p_ev_milp, soc_ev, ev_slack = self._define_ev_vars()

        # Constraints
        con_balance = self._balance_constraint(p_pv, p_fixedgen, p_bat_in, p_bat_out, p_ev_in, p_ev_out)
        con_soc_bat = self._soc_battery_constraints(soc_bat)
        con_soc_ev = self._soc_ev_constraints(soc_ev)
        con_power_limit = self._power_limit_constraints(p_pv, p_fixedgen, p_bat_in, p_bat_out, p_ev_in, p_ev_out)
        con_milp = self._milp_constraints(p_bat_milp, p_ev_milp)

        # Objective
        obj = self._create_objective_function(p_pv, p_fixedgen, p_bat_in, p_bat_out, p_ev_in, p_ev_out, soc_bat, soc_ev, ev_slack)

        # Add variables, constraints, and objective to the model
        self._add_variables_to_model(model, p_pv, p_fixedgen, p_bat_in, p_bat_out, p_bat_milp, soc_bat, p_ev_in, p_ev_out, p_ev_milp, soc_ev, ev_slack)
        self._add_constraints_to_model(model, con_balance, con_soc_bat, con_soc_ev, con_power_limit, con_milp)
        self._add_objective_to_model(model, obj)

        # Solve the model
        model.optimize()

        # Retrieve and store the optimized values
        self._store_optimized_values(p_pv, p_fixedgen, p_bat_in, p_bat_out, p_bat_milp, soc_bat, p_ev_in, p_ev_out, p_ev_milp, soc_ev, ev_slack)

    def _define_pv_power_vars(self):
        # Define decision variables for PV power
        p_pv = {}
        for plant in self._get_list_plants(plant_type="pv"):
            p_pv[plant] = [
                Var(lb=0, ub=plant_capacity) for _ in range(self.config_dict["mpc_horizon"])
            ]
        return p_pv

    def _define_fixedgen_power_vars(self):
        # Define decision variables for fixed generation power
        p_fixedgen = {}
        for plant in self._get_list_plants(plant_type="fixedgen"):
            p_fixedgen[plant] = [
                Var(lb=0, ub=plant_capacity) for _ in range(self.config_dict["mpc_horizon"])
            ]
        return p_fixedgen

    def _define_battery_vars(self):
        # Define decision variables for battery-related parameters
        p_bat_in, p_bat_out, p_bat_milp, soc_bat = {}, {}, {}, {}
        for plant in self._get_list_plants(plant_type="bat"):
            p_bat_in[plant] = [
                Var(lb=0, ub=plant_capacity) for _ in range(self.config_dict["mpc_horizon"])
            ]
            p_bat_out[plant] = [
                Var(lb=0, ub=plant_capacity) for _ in range(self.config_dict["mpc_horizon"])
            ]
            p_bat_milp[plant] = [
                Var(lb=0, ub=plant_capacity, vartype="I") for _ in range(self.config_dict["mpc_horizon"])
            ]
            soc_bat[plant] = [
                Var(lb=0, ub=1) for _ in range(self.config_dict["mpc_horizon"])
            ]
        return p_bat_in, p_bat_out, p_bat_milp, soc_bat

    def _define_ev_vars(self):
        # Define decision variables for EV-related parameters
        p_ev_in, p_ev_out, p_ev_milp, soc_ev, ev_slack = {}, {}, {}, {}, {}
        for plant in self._get_list_plants(plant_type="ev"):
            p_ev_in[plant] = [
                Var(lb=0, ub=plant_capacity) for _ in range(self.config_dict["mpc_horizon"])
            ]
            p_ev_out[plant] = [
                Var(lb=0, ub=plant_capacity) for _ in range(self.config_dict["mpc_horizon"])
            ]
            p_ev_milp[plant] = [
                Var(lb=0, ub=plant_capacity, vartype="I") for _ in range(self.config_dict["mpc_horizon"])
            ]
            soc_ev[plant] = [
                Var(lb=0, ub=1) for _ in range(self.config_dict["mpc_horizon"])
            ]
            ev_slack[plant] = [
                Var(lb=0, ub=1) for _ in range(self.config_dict["mpc_horizon"])
            ]
        return p_ev_in, p_ev_out, p_ev_milp, soc_ev, ev_slack

    def _balance_constraint(self, p_pv, p_fixedgen, p_bat_in, p_bat_out, p_ev_in, p_ev_out):
        # Create constraints to ensure power balance
        con_balance = {}
        for i in range(self.config_dict["mpc_horizon"]):
            con_balance[i] = Constraint(
                sum(p_pv[plant][i] for plant in self._get_list_plants(plant_type="pv")) +
                sum(p_fixedgen[plant][i] for plant in self._get_list_plants(plant_type="fixedgen")) +
                sum(p_bat_in[plant][i] - p_bat_out[plant][i] for plant in self._get_list_plants(plant_type="bat")) +
                sum(p_ev_in[plant][i] - p_ev_out[plant][i] for plant in self._get_list_plants(plant_type="ev")),
                lb=0, ub=0
            )
        return con_balance

    def _soc_battery_constraints(self, soc_bat):
        # Create constraints to maintain state of charge for batteries
        con_soc_bat = {}
        for plant in self._get_list_plants(plant_type="bat"):
            con_soc_bat[plant] = []
            for i in range(1, self.config_dict["mpc_horizon"]):
                con_soc_bat[plant].append(
                    Constraint(
                        soc_bat[plant][i] - soc_bat[plant][i-1] - (self.time_resolution/60) *
                        (p_bat_in[plant][i] - p_bat_out[plant][i]) / plant_capacity,
                        lb=0, ub=0
                    )
                )
        return con_soc_bat

    def _soc_ev_constraints(self, soc_ev):
        # Create constraints to maintain state of charge for EVs
        con_soc_ev = {}
        for plant in self._get_list_plants(plant_type="ev"):
            con_soc_ev[plant] = []
            for i in range(1, self.config_dict["mpc_horizon"]):
                con_soc_ev[plant].append(
                    Constraint(
                        soc_ev[plant][i] - soc_ev[plant][i-1] - (self.time_resolution/60) *
                        (p_ev_in[plant][i] - p_ev_out[plant][i]) / plant_capacity,
                        lb=0, ub=0
                    )
                )
        return con_soc_ev

    def _power_limit_constraints(self, p_pv, p_fixedgen, p_bat_in, p_bat_out, p_ev_in, p_ev_out):
        # Create constraints to limit power generation/consumption
        con_power_limit = {}
        for plant in self._get_list_plants(plant_type="pv"):
            con_power_limit[plant] = [
                Constraint(p_pv[plant][i], lb=0, ub=plant_capacity)
                for i in range(self.config_dict["mpc_horizon"])
            ]
        for plant in self._get_list_plants(plant_type="fixedgen"):
            con_power_limit[plant] = [
                Constraint(p_fixedgen[plant][i], lb=0, ub=plant_capacity)
                for i in range(self.config_dict["mpc_horizon"])
            ]
        for plant in self._get_list_plants(plant_type="bat"):
            con_power_limit[plant] = [
                Constraint(p_bat_in[plant][i] + p_bat_out[plant][i], lb=0, ub=plant_capacity)
                for i in range(self.config_dict["mpc_horizon"])
            ]
        for plant in self._get_list_plants(plant_type="ev"):
            con_power_limit[plant] = [
                Constraint(p_ev_in[plant][i] + p_ev_out[plant][i], lb=0, ub=plant_capacity)
                for i in range(self.config_dict["mpc_horizon"])
            ]
        return con_power_limit

    def _milp_constraints(self, p_bat_milp, p_ev_milp):
        # Create constraints for Mixed Integer Linear Programming (MILP)
        con_milp = {}
        for plant in self._get_list_plants(plant_type="bat"):
            con_milp[plant] = [
                Constraint(p_bat_in[plant][i] - p_bat_milp[plant][i] + p_bat_out[plant][i], lb=0, ub=0)
                for i in range(self.config_dict["mpc_horizon"])
            ]
        for plant in self._get_list_plants(plant_type="ev"):
            con_milp[plant] = [
                Constraint(p_ev_in[plant][i] - p_ev_milp[plant][i] + p_ev_out[plant][i], lb=0, ub=0)
                for i in range(self.config_dict["mpc_horizon"])
            ]
        return con_milp

    def _create_objective_function(self, p_pv, p_fixedgen, p_bat_in, p_bat_out, p_ev_in, p_ev_out, soc_bat, soc_ev, ev_slack):
        # Create the objective function
        obj = maximize(
            sum(
                self.config_dict["pv_price"][i] * p_pv[plant][i]
                for plant in self._get_list_plants(plant_type="pv")
                for i in range(self.config_dict["mpc_horizon"])
            ) +
            sum(
                self.config_dict["fixedgen_price"][i] * p_fixedgen[plant][i]
                for plant in self._get_list_plants(plant_type="fixedgen")
                for i in range(self.config_dict["mpc_horizon"])
            ) +
            sum(
                self.config_dict["bat_in_price"][i] * p_bat_in[plant][i] +
                self.config_dict["bat_out_price"][i] * p_bat_out[plant][i]
                for plant in self._get_list_plants(plant_type="bat")
                for i in range(self.config_dict["mpc_horizon"])
            ) +
            sum(
                self.config_dict["ev_in_price"][i] * p_ev_in[plant][i] +
                self.config_dict["ev_out_price"][i] * p_ev_out[plant][i]
                for plant in self._get_list_plants(plant_type="ev")
                for i in range(self.config_dict["mpc_horizon"])
            ) +
            sum(
                self.config_dict["soc_bat_price"][i] * soc_bat[plant][i]
                for plant in self._get_list_plants(plant_type="bat")
                for i in range(self.config_dict["mpc_horizon"])
            ) +
            sum(
                self.config_dict["soc_ev_price"][i] * soc_ev[plant][i]
                for plant in self._get_list_plants(plant_type="ev")
                for i in range(self.config_dict["mpc_horizon"])
            ) +
            sum(
                self.config_dict["ev_slack_price"][i] * ev_slack[plant][i]
                for plant in self._get_list_plants(plant_type="ev")
                for i in range(self.config_dict["mpc_horizon"])
            )
        )
        return obj

    def _add_variables_to_model(self, model, p_pv, p_fixedgen, p_bat_in, p_bat_out, p_bat_milp, soc_bat, p_ev_in, p_ev_out, p_ev_milp, soc_ev, ev_slack):
        # Add decision variables to the model
        for plant in self._get_list_plants(plant_type="pv"):
            model.add_var(p_pv[plant])
        for plant in self._get_list_plants(plant_type="fixedgen"):
            model.add_var(p_fixedgen[plant])
        for plant in self._get_list_plants(plant_type="bat"):
            model.add_var(p_bat_in[plant])
            model.add_var(p_bat_out[plant])
            model.add_var(p_bat_milp[plant])
            model.add_var(soc_bat[plant])
        for plant in self._get_list_plants(plant_type="ev"):
            model.add_var(p_ev_in[plant])
            model.add_var(p_ev_out[plant])
            model.add_var(p_ev_milp[plant])
            model.add_var(soc_ev[plant])
            model.add_var(ev_slack[plant])

    def _add_constraints_to_model(self, model, con_balance, con_soc_bat, con_soc_ev, con_power_limit, con_milp):
        # Add constraints to the model
        for i in range(self.config_dict["mpc_horizon"]):
            model.add_con(con_balance[i])
        for plant in self._get_list_plants(plant_type="bat"):
            model.add_con(con_soc_bat[plant])
        for plant in self._get_list_plants(plant_type="ev"):
            model.add_con(con_soc_ev[plant])
        for plant in self._get_list_plants(plant_type="pv"):
            model.add_con(con_power_limit[plant])
        for plant in self._get_list_plants(plant_type="fixedgen"):
            model.add_con(con_power_limit[plant])
        for plant in self._get_list_plants(plant_type="bat"):
            model.add_con(con_power_limit[plant])
        for plant in self._get_list_plants(plant_type="ev"):
            model.add_con(con_power_limit[plant])
        for plant in self._get_list_plants(plant_type="bat"):
            model.add_con(con_milp[plant])
        for plant in self._get_list_plants(plant_type="ev"):
            model.add_con(con_milp[plant])

    def _add_objective_to_model(self, model, obj):
        # Add the objective function to the model
        model.set_objective(obj)

    def _store_optimized_values(self, p_pv, p_fixedgen, p_bat_in, p_bat_out, p_bat_milp, soc_bat, p_ev_in, p_ev_out, p_ev_milp, soc_ev, ev_slack):
        # Store the optimized values of decision variables
        self.controller_values["p_pv"] = {
            plant: [model.get_var_sol(p_pv[plant][i]) for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="pv")
        }
        self.controller_values["p_fixedgen"] = {
            plant: [model.get_var_sol(p_fixedgen[plant][i]) for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="fixedgen")
        }
        self.controller_values["p_bat_in"] = {
            plant: [model.get_var_sol(p_bat_in[plant][i]) for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="bat")
        }
        self.controller_values["p_bat_out"] = {
            plant: [model.get_var_sol(p_bat_out[plant][i]) for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="bat")
        }
        self.controller_values["p_bat_milp"] = {
            plant: [model.get_var_sol(p_bat_milp[plant][i]) for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="bat")
        }
        self.controller_values["soc_bat"] = {
            plant: [model.get_var_sol(soc_bat[plant][i]) for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="bat")
        }
        self.controller_values["p_ev_in"] = {
            plant: [model.get_var_sol(p_ev_in[plant][i]) for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="ev")
        }
        self.controller_values["p_ev_out"] = {
            plant: [model.get_var_sol(p_ev_out[plant][i]) for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="ev")
        }
        self.controller_values["p_ev_milp"] = {
            plant: [model.get_var_sol(p_ev_milp[plant][i]) for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="ev")
        }
        self.controller_values["soc_ev"] = {
            plant: [model.get_var_sol(soc_ev[plant][i]) for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="ev")
        }
        self.controller_values["ev_slack"] = {
            plant: [model.get_var_sol(ev_slack[plant][i]) for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="ev")
        }

    def optimize(self):
        # Main optimization function
        model = Model()
        p_pv = self._define_pv_vars()
        p_fixedgen = self._define_fixedgen_vars()
        p_bat_in, p_bat_out, p_bat_milp, soc_bat = self._define_battery_vars()
        p_ev_in, p_ev_out, p_ev_milp, soc_ev, ev_slack = self._define_ev_vars()
        con_balance = self._balance_constraint(p_pv, p_fixedgen, p_bat_in, p_bat_out, p_ev_in, p_ev_out)
        con_soc_bat = self._soc_battery_constraints(soc_bat)
        con_soc_ev = self._soc_ev_constraints(soc_ev)
        con_power_limit = self._power_limit_constraints(p_pv, p_fixedgen, p_bat_in, p_bat_out, p_ev_in, p_ev_out)
        con_milp = self._milp_constraints(p_bat_milp, p_ev_milp)
        obj = self._create_objective_function(p_pv, p_fixedgen, p_bat_in, p_bat_out, p_ev_in, p_ev_out, soc_bat, soc_ev, ev_slack)
        self._add_variables_to_model(model, p_pv, p_fixedgen, p_bat_in, p_bat_out, p_bat_milp, soc_bat, p_ev_in, p_ev_out, p_ev_milp, soc_ev, ev_slack)
        self._add_constraints_to_model(model, con_balance, con_soc_bat, con_soc_ev, con_power_limit, con_milp)
        self._add_objective_to_model(model, obj)
        model.optimize()
        self._store_optimized_values(p_pv, p_fixedgen, p_bat_in, p_bat_out, p_bat_milp, soc_bat, p_ev_in, p_ev_out, p_ev_milp, soc_ev, ev_slack)


class MpcLinopy:  # The linopy implementation with subfunctions
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.controller_values = {}

    # ...

    def optimize(self):
        # Main optimization function
        lp = LinearProgram(maximize=True)
        p_pv = self._define_pv_vars_lp(lp)
        p_fixedgen = self._define_fixedgen_vars_lp(lp)
        p_bat_in, p_bat_out, p_bat_milp, soc_bat = self._define_battery_vars_lp(lp)
        p_ev_in, p_ev_out, p_ev_milp, soc_ev, ev_slack = self._define_ev_vars_lp(lp)
        con_balance = self._balance_constraint_lp(lp, p_pv, p_fixedgen, p_bat_in, p_bat_out, p_ev_in, p_ev_out)
        con_soc_bat = self._soc_battery_constraints_lp(lp, soc_bat)
        con_soc_ev = self._soc_ev_constraints_lp(lp, soc_ev)
        con_power_limit = self._power_limit_constraints_lp(lp, p_pv, p_fixedgen, p_bat_in, p_bat_out, p_ev_in, p_ev_out)
        con_milp = self._milp_constraints_lp(lp, p_bat_milp, p_ev_milp)
        obj = self._create_objective_function_lp(lp, p_pv, p_fixedgen, p_bat_in, p_bat_out, p_ev_in, p_ev_out, soc_bat, soc_ev, ev_slack)
        self._add_variables_to_model_lp(lp, p_pv, p_fixedgen, p_bat_in, p_bat_out, p_bat_milp, soc_bat, p_ev_in, p_ev_out, p_ev_milp, soc_ev, ev_slack)
        self._add_constraints_to_model_lp(lp, con_balance, con_soc_bat, con_soc_ev, con_power_limit, con_milp)
        self._add_objective_to_model_lp(lp, obj)
        lp.solve()
        self._store_optimized_values_lp(p_pv, p_fixedgen, p_bat_in, p_bat_out, p_bat_milp, soc_bat, p_ev_in, p_ev_out, p_ev_milp, soc_ev, ev_slack)

    # ...

    def _define_pv_vars_lp(self, lp):
        # Define PV power decision variables
        p_pv = {}
        for plant in self._get_list_plants(plant_type="pv"):
            p_pv[plant] = [Variable(lp, lb=0, ub=plant_capacity) for _ in range(self.config_dict["mpc_horizon"])]
        return p_pv

    def _define_fixedgen_vars_lp(self, lp):
        # Define fixed generation power decision variables
        p_fixedgen = {}
        for plant in self._get_list_plants(plant_type="fixedgen"):
            p_fixedgen[plant] = [Variable(lp, lb=0, ub=plant_capacity) for _ in range(self.config_dict["mpc_horizon"])]
        return p_fixedgen

    def _define_battery_vars_lp(self, lp):
        # Define battery power decision variables
        p_bat_in = {}
        p_bat_out = {}
        p_bat_milp = {}
        soc_bat = {}
        for plant in self._get_list_plants(plant_type="bat"):
            p_bat_in[plant] = [Variable(lp, lb=0, ub=battery_charge_rate) for _ in range(self.config_dict["mpc_horizon"])]
            p_bat_out[plant] = [Variable(lp, lb=0, ub=battery_discharge_rate) for _ in range(self.config_dict["mpc_horizon"])]
            p_bat_milp[plant] = [Variable(lp, lb=0, ub=1, vartype='integer') for _ in range(self.config_dict["mpc_horizon"])]
            soc_bat[plant] = [Variable(lp, lb=0, ub=battery_capacity) for _ in range(self.config_dict["mpc_horizon"])]
        return p_bat_in, p_bat_out, p_bat_milp, soc_bat

    def _define_ev_vars_lp(self, lp):
        # Define EV power decision variables
        p_ev_in = {}
        p_ev_out = {}
        p_ev_milp = {}
        soc_ev = {}
        ev_slack = {}
        for plant in self._get_list_plants(plant_type="ev"):
            p_ev_in[plant] = [Variable(lp, lb=0, ub=ev_charge_rate) for _ in range(self.config_dict["mpc_horizon"])]
            p_ev_out[plant] = [Variable(lp, lb=0, ub=ev_discharge_rate) for _ in range(self.config_dict["mpc_horizon"])]
            p_ev_milp[plant] = [Variable(lp, lb=0, ub=1, vartype='integer') for _ in range(self.config_dict["mpc_horizon"])]
            soc_ev[plant] = [Variable(lp, lb=0, ub=ev_capacity) for _ in range(self.config_dict["mpc_horizon"])]
            ev_slack[plant] = [Variable(lp, lb=0) for _ in range(self.config_dict["mpc_horizon"])]
        return p_ev_in, p_ev_out, p_ev_milp, soc_ev, ev_slack

    # ...

    def _balance_constraint_lp(self, lp, p_pv, p_fixedgen, p_bat_in, p_bat_out, p_ev_in, p_ev_out):
        # Create balance constraint
        con_balance = []
        for i in range(self.config_dict["mpc_horizon"]):
            balance_expr = (
                sum(p_pv[plant][i] for plant in self._get_list_plants(plant_type="pv")) +
                sum(p_fixedgen[plant][i] for plant in self._get_list_plants(plant_type="fixedgen")) +
                sum(p_bat_in[plant][i] - p_bat_out[plant][i] for plant in self._get_list_plants(plant_type="bat")) +
                sum(p_ev_in[plant][i] - p_ev_out[plant][i] for plant in self._get_list_plants(plant_type="ev"))
            )
            con_balance.append(Constraint(lp, balance_expr, "==", 0))
        return con_balance

    def _soc_battery_constraints_lp(self, lp, soc_bat):
        # Create battery state of charge constraints
        con_soc_bat = []
        for plant in self._get_list_plants(plant_type="bat"):
            for i in range(self.config_dict["mpc_horizon"] - 1):
                con_soc_bat.append(Constraint(lp, soc_bat[plant][i+1] - soc_bat[plant][i], "==", -p_bat_in[plant][i] + p_bat_out[plant][i]))
        return con_soc_bat

    def _soc_ev_constraints_lp(self, lp, soc_ev):
        # Create EV state of charge constraints
        con_soc_ev = []
        for plant in self._get_list_plants(plant_type="ev"):
            for i in range(self.config_dict["mpc_horizon"] - 1):
                con_soc_ev.append(Constraint(lp, soc_ev[plant][i+1] - soc_ev[plant][i], "==", -p_ev_in[plant][i] + p_ev_out[plant][i]))
        return con_soc_ev

    def _power_limit_constraints_lp(self, lp, p_pv, p_fixedgen, p_bat_in, p_bat_out, p_ev_in, p_ev_out):
        # Create power limit constraints
        con_power_limit = []
        for plant in self._get_list_plants(plant_type="pv"):
            for i in range(self.config_dict["mpc_horizon"]):
                con_power_limit.append(Constraint(lp, p_pv[plant][i], "<=", plant_capacity))
        for plant in self._get_list_plants(plant_type="fixedgen"):
            for i in range(self.config_dict["mpc_horizon"]):
                con_power_limit.append(Constraint(lp, p_fixedgen[plant][i], "<=", plant_capacity))
        for plant in self._get_list_plants(plant_type="bat"):
            for i in range(self.config_dict["mpc_horizon"]):
                con_power_limit.append(Constraint(lp, p_bat_in[plant][i], "<=", battery_charge_rate))
                con_power_limit.append(Constraint(lp, p_bat_out[plant][i], "<=", battery_discharge_rate))
        for plant in self._get_list_plants(plant_type="ev"):
            for i in range(self.config_dict["mpc_horizon"]):
                con_power_limit.append(Constraint(lp, p_ev_in[plant][i], "<=", ev_charge_rate))
                con_power_limit.append(Constraint(lp, p_ev_out[plant][i], "<=", ev_discharge_rate))
        return con_power_limit

    def _milp_constraints_lp(self, lp, p_bat_milp, p_ev_milp):
        # Create MILP constraints for battery and EV operation
        con_milp = []
        for plant in self._get_list_plants(plant_type="bat"):
            for i in range(self.config_dict["mpc_horizon"]):
                con_milp.append(Constraint(lp, p_bat_in[plant][i] - p_bat_out[plant][i], "<=", battery_capacity * p_bat_milp[plant][i]))
        for plant in self._get_list_plants(plant_type="ev"):
            for i in range(self.config_dict["mpc_horizon"]):
                con_milp.append(Constraint(lp, p_ev_in[plant][i] - p_ev_out[plant][i], "<=", ev_capacity * p_ev_milp[plant][i]))
        return con_milp

    def _create_objective_function_lp(self, lp, p_pv, p_fixedgen, p_bat_in, p_bat_out, p_ev_in, p_ev_out, soc_bat, soc_ev, ev_slack):
        # Create the objective function
        obj = Objective(lp)
        for plant in self._get_list_plants(plant_type="pv"):
            for i in range(self.config_dict["mpc_horizon"]):
                obj.add_term(p_pv[plant][i], pv_cost)
        for plant in self._get_list_plants(plant_type="fixedgen"):
            for i in range(self.config_dict["mpc_horizon"]):
                obj.add_term(p_fixedgen[plant][i], fixedgen_cost)
        for plant in self._get_list_plants(plant_type="bat"):
            for i in range(self.config_dict["mpc_horizon"]):
                obj.add_term(p_bat_in[plant][i] + p_bat_out[plant][i], battery_cost)
                obj.add_term(soc_bat[plant][i], battery_soc_cost)
        for plant in self._get_list_plants(plant_type="ev"):
            for i in range(self.config_dict["mpc_horizon"]):
                obj.add_term(p_ev_in[plant][i] + p_ev_out[plant][i], ev_cost)
                obj.add_term(soc_ev[plant][i], ev_soc_cost)
                obj.add_term(ev_slack[plant][i], ev_slack_cost)
        return obj

    # ...

    def _store_optimized_values_lp(self, p_pv, p_fixedgen, p_bat_in, p_bat_out, p_bat_milp, soc_bat, p_ev_in, p_ev_out, p_ev_milp, soc_ev, ev_slack):
        # Store optimized values
        self.controller_values["p_pv"] = {
            plant: [p_pv[plant][i].value() for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="pv")
        }
        self.controller_values["p_fixedgen"] = {
            plant: [p_fixedgen[plant][i].value() for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="fixedgen")
        }
        self.controller_values["p_bat_in"] = {
            plant: [p_bat_in[plant][i].value() for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="bat")
        }
        self.controller_values["p_bat_out"] = {
            plant: [p_bat_out[plant][i].value() for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="bat")
        }
        self.controller_values["p_bat_milp"] = {
            plant: [p_bat_milp[plant][i].value() for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="bat")
        }
        self.controller_values["soc_bat"] = {
            plant: [soc_bat[plant][i].value() for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="bat")
        }
        self.controller_values["p_ev_in"] = {
            plant: [p_ev_in[plant][i].value() for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="ev")
        }
        self.controller_values["p_ev_out"] = {
            plant: [p_ev_out[plant][i].value() for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="ev")
        }
        self.controller_values["p_ev_milp"] = {
            plant: [p_ev_milp[plant][i].value() for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="ev")
        }
        self.controller_values["soc_ev"] = {
            plant: [soc_ev[plant][i].value() for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="ev")
        }
        self.controller_values["ev_slack"] = {
            plant: [ev_slack[plant][i].value() for i in range(self.config_dict["mpc_horizon"])]
            for plant in self._get_list_plants(plant_type="ev")
        }
