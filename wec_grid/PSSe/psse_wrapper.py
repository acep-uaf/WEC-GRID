# Standard Libraries
import os
import sys

# 3rd Party Libraries
import pandas as pd
import cmath
import matplotlib.pyplot as plt

# local libraries
from WEC_GRID.utilities.util import read_paths
from WEC_GRID.viz.psse_viz import PSSEVisualizer

# Initialize the PATHS dictionary
PATHS = read_paths()
CURR_DIR = os.path.dirname(os.path.abspath(__file__))


class PSSeWrapper:
    psspy = None

    def __init__(self, case):
        self.case_file = case
        self.dataframe = pd.DataFrame()
        self.history = {}
        self.z_history = {}
        self.flow_data = {}
        self.wec_list = []

    def initalize(self, solver):
        """
        Description: Initializes a PSSe case, uses the topology passed at original initialization
        input:
            solver: the solver you want to use supported by PSSe, "fnsl" is a good default (str)
        output: None
        """
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        psse_path = PATHS["psse"]
        sys.path.extend(
            [
                os.path.join(psse_path, subdir)
                for subdir in ["PSSPY37", "PSSBIN", "PSSLIB", "EXAMPLE"]
            ]
        )
        os.environ["PATH"] = (
            os.path.join(psse_path, "PSSPY37")
            + ";"
            + os.path.join(psse_path, "PSSBIN")
            + ";"
            + os.path.join(psse_path, "EXAMPLE")
            + ";"
            + os.environ["PATH"]
        )

        import psse35

        psse35.set_minor(3)
        import psspy

        import dyntools

        PSSeWrapper.psspy = psspy

        psspy.report_output(islct=2, filarg="NUL", options=[0])  # Discards output

        PSSeWrapper.psspy.psseinit(50)
        self.history = {}
        self.lst_param = ["BASE", "PU", "ANGLE", "P", "Q"]
        self.solver = solver
        self.dynamic_case_file = ""

        # self.dataframe = pd.DataFrame()
        self._i = psspy.getdefaultint()
        self._f = psspy.getdefaultreal()
        self._s = psspy.getdefaultchar()

        if self.case_file.endswith(".sav"):
            ierr = PSSeWrapper.psspy.case(self.case_file)
        elif self.case_file.endswith(".raw"):
            ierr = PSSeWrapper.psspy.read(1, self.case_file)
        elif self.case_file.endswith(".RAW"):
            ierr = PSSeWrapper.psspy.read(1, self.case_file)

        if ierr >= 1:
            print("error reading")
            return 0

        if self.run_powerflow(self.solver):

            self.history[-1] = self.dataframe

            self.z_values(time=-1)

            self.store_p_flow(t=-1)
            return 1

        else:
            print("Error grabbing values from PSSe")
            return 0

    def clear(self):
        """
        Description: This function clears all the data and resets the variables of the PSSe valuables
        input: None
        output: None
        """
        # initalized variables and files
        self.lst_param = ["PU", "P", "Q"]
        self.dataframe = pd.DataFrame()
        self.history = {}
        # initialization functions
        PSSeWrapper.psspy.read(1, self.case_file)
        self.run_powerflow(self.solver)
        # program variables
        # self.history['Start'] = self.dataframe

    def run_powerflow(self, solver):
        """
        Description: This function runs the powerflow for PSSe for the given solver passed for the case in memory
        input:
             solver: the solver you want to use supported by PSSe, "fnsl" is a good default (str)
        output: None
        """
        ierr = 1  # default there is an error

        if solver == "fnsl":
            ierr = PSSeWrapper.psspy.fnsl()
        elif solver == "GS":
            ierr = PSSeWrapper.psspy.solv()
        elif solver == "DC":
            ierr = PSSeWrapper.psspy.dclf_2(1, 1, [1, 0, 1, 2, 1, 1], [0, 0, 0], "1")
        else:
            print("not a valid solver")
            return 0

        if ierr < 1:  # no error in solving
            ierr = self.get_values()
        else:
            print("Error while solving")
            return 0

        if ierr == 1:  # no error while grabbing values
            return 1
        else:
            print("Error while grabbing values")
            return 0

    def get_values(self):
        """
        Description: This function grabs all the important values we want in our dataframe
        input: None, uses lst_param tho
        output: Dataframe of the selected parameters for each bus.
        """
        lst = self.lst_param
        temp_dict = {}
        for bus_parameter in lst:
            if bus_parameter != "P" and bus_parameter != "Q":
                # grabs the bus parameter values for the specified parameter - list
                ierr, bus_parameter_values = PSSeWrapper.psspy.abusreal(
                    -1, string=bus_parameter
                )
                if ierr != 0:
                    print("error in get_values function")
                    return 0
                bus_add = {}
                for bus_index, value in enumerate(
                    bus_parameter_values[0]
                ):  # loops over those values to create bus num & value pairs
                    bus_add["BUS {}".format(bus_index + 1)] = value
                temp_dict[bus_parameter] = bus_add

        self.dataframe = pd.DataFrame.from_dict(temp_dict)
        self.dataframe = self.dataframe.reset_index()
        self.dataframe = self.dataframe.rename(columns={"index": "Bus"})
        # gets the bus type (3 = swing)
        self.dataframe["Type"] = PSSeWrapper.psspy.abusint(-1, string="TYPE")[1][0]
        self.dataframe.insert(0, "BUS_ID", range(1, 1 + len(self.dataframe)))
        self.addGeninfo()
        self.addLoadinfo()

        if "P" in lst:
            self.get_p_or_q("P")
        if "Q" in lst:
            self.get_p_or_q("Q")

        # Check if column exists, if not then initialize
        if "ΔP" not in self.dataframe.columns:
            self.dataframe["ΔP"] = 0.0  # default value
        if "ΔQ" not in self.dataframe.columns:
            self.dataframe["ΔQ"] = 0.0  # default value
        if "M_Angle" not in self.dataframe.columns:
            self.dataframe["M_Angle"] = 0.0  # default value
        if "M_Mag" not in self.dataframe.columns:
            self.dataframe["M_Mag"] = 0.0  # default value

        # Your loop remains unchanged
        for index, row in self.dataframe.iterrows():
            mismatch = PSSeWrapper.psspy.busmsm(row["BUS_ID"])[1]
            self.dataframe.at[index, "ΔP"] = mismatch.real  # should be near zero
            self.dataframe.at[index, "ΔQ"] = mismatch.imag
            self.dataframe.at[index, "M_Angle"] = abs(mismatch)
            self.dataframe.at[index, "M_Mag"] = cmath.phase(mismatch)
        return 1

    def get_p_or_q(self, letter):
        """
        Description: retrieves P (activate) or Q (reactive) Voltage (in PU) and Voltage Angle for each Bus in the current loaded case
        input:
            letter: either P or Q as a string
        output: None
        """
        gen_values = self.dataframe["{} Gen".format(letter)]  #
        load_values = self.dataframe["{} Load".format(letter)]
        letter_list = [None] * len(self.dataframe)

        for i in range(len(letter_list)):
            gen = gen_values[i]
            load = load_values[i]
            if (not pd.isnull(gen)) and (not pd.isnull(load)):
                letter_list[i] = gen - load
            else:
                if not pd.isnull(gen):
                    letter_list[i] = gen
                if not pd.isnull(load):
                    letter_list[i] = 0 - load  # gen is
        self.dataframe["{}".format(letter)] = letter_list

    def busNum(self):
        """
        Description: Returns the number of Buses in the currently loaded case
        input: None
        output: Number of Buses
        """
        PSSeWrapper.psspy.bsys(0, 0, [0.0, 0.0], 1, [1], 0, [], 0, [], 0, [])
        ierr, all_bus = PSSeWrapper.psspy.abusint(0, 1, ["number"])
        return all_bus[0]

    def addGeninfo(self):
        """
        Description: This function grabs the generator values from the PSSe system and updates the dataframe with the generator data.
        Input: None
        Output: None but updates dataframe with generator data
        """
        machine_bus_nums = PSSeWrapper.psspy.amachint(-1, 4, "NUMBER")[1][
            0
        ]  # get the bus numbers of the machines - list
        # grabs the complex values for the machine
        ierr, machine_bus_values = PSSeWrapper.psspy.amachcplx(-1, 1, "PQGEN")
        if ierr != 0:
            raise Exception("Error in grabbing PGGEN values in addgen function")
        p_gen_df_list = [None] * len(self.dataframe)
        q_gen_df_list = [None] * len(self.dataframe)
        # iterate over the machine values
        for list_index, value in enumerate(machine_bus_values[0]):
            p_gen_df_list[
                machine_bus_nums[list_index] - 1
            ] = value.real  # -1 is for the offset
            q_gen_df_list[machine_bus_nums[list_index] - 1] = value.imag

        self.dataframe["P Gen"] = p_gen_df_list
        self.dataframe["Q Gen"] = q_gen_df_list

    def addLoadinfo(self):
        """
        Description: this function grabs the load values from the PSSe system
        input: None
        output: None but updates dataframe with load data
        """
        load_bus_nums = PSSeWrapper.psspy.aloadint(-1, 4, "NUMBER")[1][
            0
        ]  # get the bus numbers of buses with loads - list
        ierr, load_bus_values = PSSeWrapper.psspy.aloadcplx(
            -1, 1, "MVAACT"
        )  # load values
        if ierr != 0:
            raise Exception("Error in grabbing PGGEN values in addgen function")
        p_load_df_list = [None] * len(self.dataframe)
        q_load_df_list = [None] * len(self.dataframe)

        # iterate over the machine values
        for list_index, value in enumerate(load_bus_values[0]):
            p_load_df_list[
                load_bus_nums[list_index] - 1
            ] = value.real  # -1 is for the offset
            q_load_df_list[load_bus_nums[list_index] - 1] = value.imag
        self.dataframe["P Load"] = p_load_df_list
        self.dataframe["Q Load"] = q_load_df_list

    def z_values(self, time):
        """
        Retrieve the impedance values for each branch in the power grid at a given time.

        Parameters:
        - time (float): The time at which to retrieve the impedance values.

        Returns:
        - None. The impedance values are stored in the `z_history` attribute of the object.
        """
        # Retrieve FROMNUMBER and TONUMBER for all branches
        ierr, (from_numbers, to_numbers) = PSSeWrapper.psspy.abrnint(
            sid=-1, flag=3, string=["FROMNUMBER", "TONUMBER"]
        )
        assert ierr == 0, "Error retrieving branch data"

        # Create a dictionary to store the impedance values for each branch
        impedances = {}

        for from_bus, to_bus in zip(from_numbers, to_numbers):
            ickt = "1"  # Assuming a default circuit identifier; might need adjustment for your system

            ierr, cmpval = PSSeWrapper.psspy.brndt2(from_bus, to_bus, ickt, "RX")
            if ierr == 0:
                impedances[
                    (from_bus, to_bus)
                ] = cmpval  # Store the complex impedance value directly
            else:
                print(f"Error fetching impedance data for branch {from_bus}-{to_bus}")

        # The impedances dictionary contains impedance for each branch
        self.z_history[time] = impedances

    def store_p_flow(self, t):
        """
        Function to store the p_flow values of a grid network in a dictionary.

        Parameters:
        - t (float): Time at which the p_flow values are to be retrieved.
        """
        # Create an empty dictionary for this particular time
        p_flow_dict = {}

        try:
            ierr, (fromnumber, tonumber) = PSSeWrapper.psspy.abrnint(
                sid=-1, flag=3, string=["FROMNUMBER", "TONUMBER"]
            )

            for index in range(len(fromnumber)):
                ierr, p_flow = PSSeWrapper.psspy.brnmsc(
                    int(fromnumber[index]), int(tonumber[index]), "1", "P"
                )

                source = str(fromnumber[index]) if p_flow >= 0 else str(tonumber[index])
                target = str(tonumber[index]) if p_flow >= 0 else str(fromnumber[index])

                p_flow_dict[(source, target)] = p_flow

            # Store the p_flow data for this time in the flow_data dictionary
            self.flow_data[t] = p_flow_dict

        except Exception as e:
            print(f"Error fetching data: {e}")

    def update_type(self):
        for wec in self.wec_list:
            self.dataframe.loc[self.dataframe["BUS_ID"] == wec.bus_location, "Type"] = 4

    def run_dynamics(self, dyr_file=""):
        import numpy as np

        """
        Description: This function is a wrapper around the dynamic modeling process in PSSe, this function checks for .dyr file and
        then proceeds to run a simple simulations.
        input: you can add dyr file path or the function will just ask you via CL
        output: None
        """

        output_file = "./simulation_output_file.out"
        solved_case = "solved_case.sav"

        # check if dynamic file is loaded
        if self.dynamic_case_file == "":
            self.c = input("Dynamic File location")

        PSSeWrapper.psspy.dyre_new(dyrefile=dyr_file)
        PSSeWrapper.psspy.fnsl()
        PSSeWrapper.psspy.save(solved_case)

        # Setup for dynamic simulation
        PSSeWrapper.psspy.cong(0)
        PSSeWrapper.psspy.ordr(0)
        PSSeWrapper.psspy.fact()
        PSSeWrapper.psspy.tysl(0)

        # Add channels (e.g., bus voltage, machine speed)
        PSSeWrapper.psspy.chsb(
            sid=0, all=1, status=[-1, -1, -1, 1, 13, 0]
        )  # Bus voltage
        PSSeWrapper.psspy.chsb(
            sid=0, all=1, status=[-1, -1, -1, 1, 7, 0]
        )  # Machine speed
        PSSeWrapper.psspy.chsb(sid=0, all=1, status=[-1, -1, -1, 1, 1, 0])  # ANGLE
        PSSeWrapper.psspy.chsb(sid=0, all=1, status=[-1, -1, -1, 1, 12, 0])  # bus freq

        # Start recording to output file and run dynamic simulation
        PSSeWrapper.psspy.strt(outfile=output_file)
        end_time = 10  #
        time_step = 0.01 / end_time  # Half-cycle step
        for current_time in np.arange(0, end_time, time_step):
            PSSeWrapper.psspy.run(tpause=current_time)
            # if current_time == 1.0:
            #     psspy.dist_branch_trip(1, 5, '1')
            # if current_time == 5.0:
            #     psspy.dist_branch_close(1, 5,'1')

        PSSeWrapper.psspy.save("final_state_after_dynamic_sim.sav")

        # # Convert loads (3 step process):
        # PSSeWrapper.psspy.conl(-1, 1, 1)

        # PSSeWrapper.psspy.conl(
        #     sid=-1, all=1, apiopt=2, status=[0, 0], loadin=[100, 0, 0, 100]
        # )

        # PSSeWrapper.psspy.conl(-1, 1, 3)

        # # Convert generators:
        # PSSeWrapper.psspy.cong()

        # # Solve for dynamics
        # PSSeWrapper.psspy.ordr()
        # PSSeWrapper.psspy.fact()
        # PSSeWrapper.psspy.tysl()
        # # Save converted case
        # case_root = os.path.splitext(self.case_file)[0]
        # PSSeWrapper.psspy.save(case_root + ".sav")

        # PSSeWrapper.psspy.dyre_new(dyrefile=self.dynamic_case_file)

        # # Add channels by subsystem
        # #   BUS VOLTAGE
        # PSSeWrapper.psspy.chsb(sid=0, all=1, status=[-1, -1, -1, 1, 13, 0])
        # #   MACHINE SPEED
        # PSSeWrapper.psspy.chsb(sid=0, all=1, status=[-1, -1, -1, 1, 7, 0])

        # # Add channels individually
        # #   BRANCH MVA
        # # psspy.branch_mva_channel([-1,-1,-1,3001,3002],'1')

        # path = os.path.abspath(os.path.dirname(self.case_file)) + "\\test.snp"
        # # Save snapshot
        # PSSeWrapper.psspy.snap(sfile=path)

        # # Initialize
        # PSSeWrapper.psspy.strt(outfile=path)

        # # Run to 3 cycles
        # time = 3.0 / 60.0
        # PSSeWrapper.psspy.run(tpause=time)

    def plot_simulation_results(self, output_file, channels):
        import dyntools
        import numpy as np

        chnf_obj = dyntools.CHNF(output_file)
        _, channel_ids, channel_data = chnf_obj.get_data()
        base_frequency = 60.0  # Hz

        plt.figure(figsize=(10, 6))

        # Assuming channels for generator frequencies are correctly numbered

        for ch in channels:
            frequency_deviation = np.array(channel_data[ch])
            time_array = np.array(channel_data["time"])
            actual_frequency = 60.0 * (1 + frequency_deviation)

            plt.plot(time_array, actual_frequency, label=f"Channel {ch} Frequency")

        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Generator Frequency Over Time")
        plt.legend()
        plt.grid(True)
        plt.ylim(59.96, 60.012)
        plt.show()
        return plt

    def update_load(self, ibus, time_step):
        """
        Update the load at the given bus using the load_profiles DataFrame for the given time_step.

        Parameters:
        - ibus (int): The bus number to update.
        - time_step (int): The time step index in the load_profiles DataFrame.
        - load_profiles_df (pd.DataFrame): The load profile DataFrame. Columns should be bus numbers and rows are the time steps.

        Returns:
        - int: Error code from PSS/E API call.
        """

        load_value = self.load_profiles.loc[
            self.load_profiles["time"] == time_step, f"bus {ibus}"
        ].values[0]

        # Default values from the documentation
        _id = "1"  # Identifier; default is '1' for most loads
        # intgar = [1, 0, 0, 0, 1, 0, 0]  # default values based on documentation
        realar = [
            load_value,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]  # Update active power, keep others default
        lodtyp = "CONSTP"  # Load type description; 'CONSTP' for constant power
        intgar = [1, 1, 1, 1, 1, 0, 0]

        # Call PSS/E API to update load
        ierr = PSSeWrapper.psspy.load_data_6(ibus, _id, intgar, realar, lodtyp)

        return ierr

    def steady_state(self, gen, load, time, solver):

        for bus, gen_values in gen.items():
            pg = gen_values[0]  # pg
            ierr = PSSeWrapper.psspy.machine_data_4(
                bus, "1", realar1=pg, realar3=gen_values[2], realar4=gen_values[3]
            )  # adjust activate power
            if ierr > 0:
                raise Exception("Error in AC injection")
            vs = gen_values[1]  # vs
            ierr = PSSeWrapper.psspy.bus_chng_4(
                bus, 0, realar2=vs
            )  # adjsut voltage mag PU
            if ierr > 0:
                raise Exception("Error in AC injection")
            print("gen at bus {} is {} at time {}".format(bus, pg, time))

        for bus, load_value in load.items():
            ierr = PSSeWrapper.psspy.load_data_6(
                bus, "1", realar1=load_value["P"], realar2=load_value["Q"]
            )

        self.run_powerflow(solver)
        self.update_type()
        self.history[time] = self.dataframe

    def ac_injection(self, start, end, p=None, v=None, time=None):
        """
        Description: WEC AC injection for PSSe powerflow solver
        input:
            p - a vector of active power values in order of bus num
            v - a vector of voltage mag PU values in order of bus num
            pf_solver - Power flow solving algorithm  (Default-"fnsl")
            time: (Int)
        output:
            no output but dataframe is updated and so is history
        """
        num_wecs = len(self.wec_list)
        num_cecs = len(self.cec_list)

        time = self.wec_list[0].dataframe.time.to_list()
        for t in time:
            # print("time: {}".format(t))
            if t >= start and t <= end:
                if num_wecs > 0:
                    for idx, wec_obj in enumerate(self.wec_list):
                        bus = wec_obj.bus_location
                        pg = wec_obj.dataframe.loc[
                            wec_obj.dataframe.time == t
                        ].pg  # adjust activate power
                        ierr = PSSeWrapper.psspy.machine_data_2(
                            bus, "1", realar1=pg
                        )  # adjust activate power
                        if ierr > 0:
                            raise Exception("Error in AC injection")
                        vs = wec_obj.dataframe.loc[wec_obj.dataframe.time == t].vs
                        ierr = PSSeWrapper.psspy.bus_chng_4(
                            bus, 0, realar2=vs
                        )  # adjsut voltage mag PU
                        if ierr > 0:
                            raise Exception("Error in AC injection")

                        # self.run_powerflow(self.solver)
                        self.update_load(bus, t)
                        # print("=======")plot_bus
                if num_cecs > 0:
                    if t in self.cec_list[0].dataframe["time"].values:
                        for idx, cec_obj in enumerate(self.cec_list):
                            bus = cec_obj.bus_location
                            pg = cec_obj.dataframe.loc[
                                cec_obj.dataframe.time == t
                            ].pg  # adjust activate power
                            ierr = PSSeWrapper.psspy.machine_data_2(
                                bus, "1", realar1=pg
                            )  # adjust activate power
                            if ierr > 0:
                                raise Exception("Error in AC injection")
                            vs = wec_obj.dataframe.loc[wec_obj.dataframe.time == t].vs
                            ierr = PSSeWrapper.psspy.bus_chng_4(
                                bus, 0, realar2=vs
                            )  # adjsut voltage mag PU
                            if ierr > 0:
                                raise Exception("Error in AC injection")

                            # self.run_powerflow(self.solver)
                            self.update_load(bus, t)
                        #     #print("=======")

                self.run_powerflow(self.solver)
                self.update_type()
                self.history[t] = self.dataframe
                self.z_values(time=t)
                self.store_p_flow(t)
            if t > end:
                break
        return

    def bus_history(self, bus_num):
        """
        Description: this function grab all the data associated with a bus through the simulation
        input:
            bus_num: bus number (Int)
        output:
            bus_dataframe: a pandas dateframe of the history
        """
        # maybe I should add a filering parameter?

        bus_dataframe = pd.DataFrame()
        for time, df in self.history.items():
            temp = pd.DataFrame(df.loc[df["BUS_ID"] == bus_num])
            temp.insert(0, "time", time)
            bus_dataframe = bus_dataframe.append(temp)
        return bus_dataframe

    # def plot_bus(self, bus_num, time, arg_1="P", arg_2="Q"):
    #     """
    #     Description: This function plots the activate and reactive power for a given bus
    #     input:
    #         bus_num: the bus number we wanna viz (Int)
    #         time: a list with start and end time (list of Ints)
    #     output:
    #         matplotlib chart
    #     """
    #     visualizer = PSSEVisualizer(
    #         dataframe=self.dataframe,
    #         history=self.history,
    #         #load_profiles=self.load_profiles,
    #         #flow_data=self.get_flow_data(),
    #     )

    def plot_bus(self, bus_num, time, arg_1="P", arg_2="Q"):
        """
        Description: This function plots the activate and reactive power for a given bus
        input:
            bus_num: the bus number we wanna viz (Int)
            time: a list with start and end time (list of Ints)
        output:
            matplotlib chart
        """
        visualizer = PSSEVisualizer(
            psse_dataframe=self.dataframe,
            psse_history=self.history,
        )
        visualizer.plot_bus(bus_num, time, arg_1, arg_2)

    def plot_load_curve(self, bus_id):
        """
        Description: This function plots the load curve for a given bus
        input:
            bus_id: the bus number we want to visualize (Int)
        output:
            matplotlib chart
        """
        # Check if the bus_id exists in load_profiles
        viz = PSSEVisualizer(
            dataframe=self.dataframe,
            history=self.history,
            # load_profiles=self.load_profiles,
            # flow_data=self.flow_data,
        )
        viz.plot_load_curve(bus_id)

    def viz(self, dataframe=None):
        """
        Description: Generates a visualization of the PSSE data using the PSSEVisualizer class.

        Parameters:
        - dataframe (pandas.DataFrame): Optional parameter to pass a custom PSSE dataframe.

        Returns:
        - matplotlib.figure.Figure: A matplotlib figure object containing the visualization.
        """
        visualizer = PSSEVisualizer(
            dataframe=self.dataframe,
            history=self.history,
            # load_profiles=self.load_profiles,
            # flow_data=self.get_flow_data(),
        )
        return visualizer.viz()

    def adjust_gen(self, bus_num, p=None, v=None, q=None):
        """
        Description: Given a generator bus number. Adjust the values based on the parameters passed.
        input:
        output:
        """

        if p is not None:
            ierr = PSSeWrapper.psspy.machine_data_2(
                bus_num, "1", realar1=p
            )  # adjust activate power
            if ierr > 0:
                raise Exception("Error in AC injection")

        if v is not None:
            ierr = PSSeWrapper.psspy.bus_chng_4(
                bus_num, 0, realar2=v
            )  # adjsut voltage mag PU
            if ierr > 0:
                raise Exception("Error in AC injection")

        if q is not None:
            ierr = PSSeWrapper.psspy.machine_data_2(
                bus_num, "1", realar2=q
            )  # adjust Reactivate power
            if ierr > 0:
                raise Exception("Error in AC injection")

    def get_flow_data(self, t=None):
        """
        Description:
        This method retrieves the power flow data for all branches in the power system at a given timestamp.
        If no timestamp is provided, the method fetches the data from PSS/E and returns it.
        If a timestamp is provided, the method retrieves the corresponding data from the dictionary and returns it.

        Inputs:
        - t (float): timestamp for which to retrieve the power flow data (optional)

        Outputs:
        - flow_data (dict): dictionary containing the power flow data for all branches in the power system
        """
        # If t is not provided, fetch data from PSS/E
        if t is None:
            flow_data = {}

            try:
                ierr, (fromnumber, tonumber) = self.psspy.abrnint(
                    sid=-1, flag=3, string=["FROMNUMBER", "TONUMBER"]
                )

                for index in range(len(fromnumber)):
                    ierr, p_flow = self.psspy.brnmsc(
                        int(fromnumber[index]), int(tonumber[index]), "1", "P"
                    )

                    edge_data = {
                        "source": str(fromnumber[index])
                        if p_flow >= 0
                        else str(tonumber[index]),
                        "target": str(tonumber[index])
                        if p_flow >= 0
                        else str(fromnumber[index]),
                        "p_flow": p_flow,
                    }

                    # Use a tuple (source, target) as a unique identifier for each edge
                    edge_identifier = (edge_data["source"], edge_data["target"])
                    flow_data[edge_identifier] = edge_data["p_flow"]
            except Exception as e:
                print(f"Error fetching data: {e}")

            # Assign the fetched data to the current timestamp and return it
            # self.flow_data[time.time()] = flow_data
            return flow_data

        # If t is provided, retrieve the corresponding data from the dictionary
        else:
            return self.flow_data.get(t, {})

    def _psse_dc_injection(self, ibus, p, pf_solver, time):
        """
        Description: preforms the DC injection of the wec buses
        input:
            p: a list of active power set point in order(list)
            pf_solver: supported PSSe solver (Str)
            time: (Int)
        output: None
        """
        ierr = Wec_grid.psspy.machine_chng_3(ibus, "1", [], [p])
        if ierr > 0:
            print("Failed | machine_chng_3 code = {}".format(ierr))
        # psspy.dclf_2(status4=2)
        ierr = Wec_grid.psspy.dclf_2(1, 1, [1, 0, 1, 2, 0, 1], [0, 0, 1], "1")
        if ierr > 0:
            raise Exception("Error in DC injection")
        self._psse_get_values()
        self.history[time] = self.dataframe