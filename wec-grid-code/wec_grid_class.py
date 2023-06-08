# Wec_grid_class 
import os
import sys

paths = []
with open('../wec-grid-code/path_config.txt', 'r') as fp:
    while 1:
        line = fp.readline()
        if len(line) == 0:  # end of file break
            break
        temp = line.split('\n')
        paths.append(r'{}'.format(temp[0])) #will allow spaces in path strings

psse_path = paths[0]
wec_sim_path = paths[1]
wec_model_path = paths[2]
wec_grid_class_path = paths[3]
wec_grid_folder = paths[4]

# Path stuff

sys.path.append(psse_path + "\\PSSPY37")
sys.path.append(psse_path + "\\PSSBIN")
sys.path.append(psse_path + "\\PSSLIB")
sys.path.append(psse_path + "\\EXAMPLE")
sys.path.append(psse_path + "\\PSSPY37")
os.environ['PATH'] = (
            psse_path + "\\PSSPY37;" + psse_path + "\\PSSBIN;" + psse_path + "\\EXAMPLE;" + os.environ['PATH'])

import pandas as pd
import psse35

psse35.set_minor(3)
import matplotlib.pyplot as plt
import psspy
import redirect

redirect.psse2py()


class Wec_grid:
    def __init__(self, case, solver, wec_bus):
        """
        Descrtiption:
        input:
        output
        """
        psspy.psseinit(50)
        # initalized variables and files
        self.case_file = case
        self.dynamic_case_file = ""
        self.lst_param = ['BASE', 'PU', 'ANGLED', 'P', 'Q']
        self.dataframe = pd.DataFrame()
        self.wecBus_num = wec_bus
        self.history = {}
        self.solver = solver
        self._i = psspy.getdefaultint()
        self._f = psspy.getdefaultreal()
        self._s = psspy.getdefaultchar()

        # initialization functions

        if self.case_file.endswith('.sav'):
            psspy.case(self.case_file)
        elif self.case_file.endswith('.raw'):
            psspy.read(1, self.case_file)
        elif self.case_file.endswith('.RAW'):
            psspy.read(1, self.case_file)

        self.run_powerflow(self.solver)

        # program variables
        self.swingBus = self.dataframe.loc[self.dataframe['Bus'] == 'BUS 1']
        self.swingBus.insert(0, 'time', None)
        self.wecBus = self.dataframe.loc[self.dataframe['Bus'] == 'BUS {}'.format(str(self.wecBus_num))]
        self.wecBus.insert(0, 'time', None)
        self.history['Start'] = self.dataframe

    def clear(self):
        """
        Descrtiption:
        input:
        output:
        """

        # initalized variables and files
        self.lst_param = ['BASE', 'PU', 'ANGLED', 'P', 'Q']
        self.dataframe = pd.DataFrame()
        self.history = {}

        # initialization functions
        psspy.read(1, self.case_file)
        self.run_powerflow(self.solver)

        # program variables
        self.swingBus = self.dataframe.loc[self.dataframe['Bus'] == 'BUS 1']
        self.swingBus.insert(0, 'time', None)
        self.wecBus = self.dataframe.loc[self.dataframe['Bus'] == 'BUS {}'.format(str(self.wecBus_num))]
        self.wecBus.insert(0, 'time', None)
        self.history['Start'] = self.dataframe

    def run_dynamics(self):
        """
        Descrtiption:
        input:
        output:
        """

        # check if dynamic file is loaded
        if self.dynamic_case_file == "":
            self.dynamic_case_file = input("Dynamic File location")
        # self.dynamic_case_file = "../input_files/dynamics.dyr"
        # Convert Generators
        psspy.cong()
        # Solve for dynamics
        psspy.ordr()
        psspy.fact()
        psspy.tysl()
        # Add Dynamics data file
        psspy.dyre_new(dyrefile=self.dynamic_case_file)
        # Add channels and parameters
        # BUS VOLTAGE
        psspy.chsb(sid=0, all=1, status=[-1, -1, -1, 1, 13, 0])
        # Active and Reactive Power Flow
        psspy.chsb(sid=0, all=1, status=[-1, -1, -1, 1, 16, 0])

        # Save snapshoot
        path = wec_grid_folder + "\\output_files\\test.out"
        psspy.snap(sfile=path)

        # Initialize
        psspy.strt(outfile=path)

        psspy.strt()
        psspy.run(tpause=1)
        # Set simulation parameter step size
        psspy.dynamics_solution_params(realar3=0.01)

    def run_WEC_Sim(self):
        """
        Descrtiption:
        input:
        output:
        """
        # TODO add simulation time arg pass
        import matlab.engine
        eng = matlab.engine.start_matlab()
        print("Matlab Engine estbalished")
        eng.cd(wec_model_path)
        path = wec_sim_path  # Update to match your WEC-SIM source location
        eng.addpath(eng.genpath(path), nargout=0)
        
        # Variables required to run w2gSim
        print("calling W2G")
        eng.workspace['wecId'] = 1
        eng.workspace['simLength'] = 3600
        eng.workspace['Tsample'] = 300
        eng.workspace['waveHeight'] = 2.5
        eng.workspace['wavePeriod'] = 8
        eng.workspace['waveSeed'] = np.random.randint(999999999)
        eng.eval("m2g_out = w2gSim(wecId,simLength,Tsample,waveHeight,wavePeriod,waveSeed);", nargout=0)
        print("displaying simulation plots")
        display(Image(filename="..\input_files\W2G_RM3\sim_figures\Pgen_Pgrid_Qgrid.jpg"))
        display(Image(filename="..\input_files\W2G_RM3\sim_figures\Pgen_Pgrid_comp.jpg"))
        display(Image(filename="..\input_files\W2G_RM3\sim_figures\DClink_voltage.jpg"))
        print("calling PSSe formatting")
        eng.eval("WECsim_to_PSSe_dataFormatter",nargout=0)
        print("sim complete")

    def run_powerflow(self, solver):
        if solver == 'fnsl':
            psspy.fnsl()
            self.get_values()
        elif solver == 'GS':
            psspy.solv()
            self.get_values()
        elif solver == 'DC':
            psspy.dclf_2(1, 1, [1, 0, 1, 2, 1, 1], [0, 0, 0], '1')
            self.get_values()
        else:
            print("error in run_pf")

    def get_values(self):
        """
        Descrtiption: 
        input: List of Parameters such as BASE, PU, KV, ANGLED 
        output: Dataframe of the selected parameters for each bus
        """

        lst = self.lst_param
        temp_dict = {}
        for bus_parameter in lst:
            if bus_parameter != "P" and bus_parameter != "Q":
                ierr, bus_parameter_values = psspy.abusreal(-1, string=bus_parameter)  # grabs the bus parameter values for the specified parameter - list
                if ierr != 0:
                    print("error in get_values function")
                bus_add = {}
                for bus_index, value in enumerate(
                        bus_parameter_values[0]):  # loops over those values to create bus num & value pairs
                    bus_add['BUS {}'.format(bus_index + 1)] = value
                temp_dict[bus_parameter] = bus_add

        self.dataframe = pd.DataFrame.from_dict(temp_dict)
        self.dataframe = self.dataframe.reset_index()
        self.dataframe = self.dataframe.rename(columns={'index': "Bus"})
        self.dataframe['Type'] = psspy.abusint(-1, string="TYPE")[1][0]  # gets the bus type (3 = swing)
        self.dataframe.insert(0, "BUS_ID", range(1, 1 + len(self.dataframe)))
        self.addGeninfo()
        self.addLoadinfo()

        if "P" in lst:
            self.get_p_or_q('P')
        if "Q" in lst:
            self.get_p_or_q('Q')

    def get_p_or_q(self, letter):
        """
        Descrtiption: retre P (activate) Q (reactive) Voltage (in PU) and Voltage Angle for each Bus in the current loaded case
        input:
        output:
        """
        gen_values = self.dataframe['{} Gen'.format(letter)]  #
        load_values = self.dataframe['{} Load'.format(letter)]
        letter_list = [None] * len(self.dataframe)

        for i in range(len(letter_list)):
            gen = gen_values[i]
            load = load_values[i]

            if (not pd.isnull(gen)) and (not pd.isnull(load)):
                letter_list[i] = gen - load
            else:
                if (not pd.isnull(gen)):
                    letter_list[i] = gen
                if (not pd.isnull(load)):
                    letter_list[i] = 0 - load  # gen is

        self.dataframe['{}'.format(letter)] = letter_list

    def busNum(self):
        """
        Descrtiption: Returns the number of Buses in the currently loaded case
        input: None 
        output: Number of Buses
        """
        psspy.bsys(0, 0, [0.0, 0.0], 1, [1], 0, [], 0, [], 0, [])
        ierr, all_bus = psspy.abusint(0, 1, ['number'])
        return all_bus[0]

    def dc_injection(self, ibus, p, pf_solver, time):
        """
        Descrtiption:
        input:
        output:
        """
        ierr = psspy.machine_chng_3(ibus, "1", [], [p])
        if ierr > 0:
            print("Failed | machine_chng_3 code = {}".format(ierr))
        psspy.dclf()
        self.get_values()

        temp = pd.DataFrame(self.dataframe.loc[self.dataframe['Bus'] == 'BUS 1'])
        temp.insert(0, 'time', time)
        self.swingBus = self.swingBus.append(temp)

        temp = pd.DataFrame(self.dataframe.loc[self.dataframe['Bus'] == 'BUS {}'.format(str(self.wecBus_num))])
        temp.insert(0, 'time', time)
        self.wecBus = self.wecBus.append(temp)

        self.history[time] = self.dataframe

    def ac_injection(self, ibuses, p, v, pf_solver, time):
        """
        Descrtiption:
        input:
        output:
        """
        # for list_index, bus in enumerate(ibuses):
        #     ierr = psspy.machine_chng_3(bus, "1", [], [p[list_index]])
        #     if ierr > 0:
        #         print("Failed | machine_chng_3 code = {}".format(ierr))
        #
        #     ierr = psspy.bus_chng_4(bus, 0, [], [self._f, v[list_index]])
        #     if ierr > 0:
        #         print("Failed | bus_chng_4 code = {}".format(ierr))
        for idx, bus in enumerate(ibuses):
            ierr = psspy.machine_data_2(bus, '1', realar1=p[idx])
            if ierr > 0:
                print("Failed | machine_chng_3 code = {}".format(ierr))

            ierr = psspy.bus_chng_4(bus, 0, realar2=v[idx])
            if ierr > 0:
                print("Failed | bus_chng_4 code = {}".format(ierr))

        self.run_powerflow(pf_solver)
        temp = pd.DataFrame(self.dataframe.loc[self.dataframe['Bus'] == 'BUS 1'])
        temp.insert(0, 'time', time)
        self.swingBus = self.swingBus.append(temp)
        temp = pd.DataFrame(self.dataframe.loc[self.dataframe['Bus'] == 'BUS {}'.format(str(self.wecBus_num))])
        temp.insert(0, 'time', time)
        self.wecBus = self.wecBus.append(temp)

        self.history[time] = self.dataframe

    # def plotSwingBus(self):
    #     """
    #     Descrtiption:
    #     input:
    #     output:
    #     """
    #     # fig, axs = plt.subplots(2)
    #     # plt.plot(self.swingBus.time, self.swingBus[letter], marker="o", markersize=5, markerfacecolor="green")
    #     # plt.xlabel("Time (sec)")
    #     # plt.ylabel("{} in MW".format(letter))
    #     # plt.title("Swing bus")
    #     fig, (ax1, ax2) = plt.subplots(2)
    #     fig.suptitle("Swing bus")
    #     ax1.plot(self.swingBus.time, self.swingBus["P"], marker="o", markersize=5, markerfacecolor="green")
    #     ax2.plot(self.swingBus.time, self.swingBus["Q"], marker="o", markersize=5, markerfacecolor="green")
    #     ax1.set(xlabel="Time(sec)", ylabel="P(MW)")
    #     ax2.set(xlabel="Time(sec)", ylabel="Q(MW)")
    #     plt.show()
    #
    # def plotWecBus(self, mode="Gen"):
    #     """
    #     Descrtiption:
    #     input:
    #     output:
    #     """
    #     if mode == "Gen":
    #         fig, (ax1, ax2) = plt.subplots(2)
    #         fig.suptitle("Swing bus")
    #         ax1.plot(self.wecBus.time, self.wecBus["P"], marker="o", markersize=5, markerfacecolor="green")
    #         ax2.plot(self.wecBus.time, self.wecBus["Q"], marker="o", markersize=5, markerfacecolor="green")
    #         ax1.set(xlabel="Time(sec)", ylabel="P (MW)")
    #         ax2.set(xlabel="Time(sec)", ylabel="Q (sMW)")
    #         plt.show()
    #     else:
    #         fig, (ax1, ax2) = plt.subplots(2)
    #         fig.suptitle("Swing bus")
    #         ax1.plot(self.wecBus.time, self.wecBus["P Gen"], marker="o", markersize=5, markerfacecolor="green")
    #         ax2.plot(self.wecBus.time, self.wecBus["Q Gen "], marker="o", markersize=5, markerfacecolor="green")
    #         ax1.set(xlabel="Time(sec)", ylabel="P Gen (MW)")
    #         ax2.set(xlabel="Time(sec)", ylabel="Q Gen (MW)")

    def addGeninfo(self):
        """
        Descrtiption:
        input:
        output:
        """
        machine_bus_nums = psspy.amachint(-1, 4, "NUMBER")[1][0]  # get the bus numbers of the machines - list
        ierr, machine_bus_values = psspy.amachcplx(-1, 1, 'PQGEN')  # grabs the complex values for the machine
        if ierr != 0:
            raise Exception('Error in grabbing PGGEN values in addgen function')
        p_gen_df_list = [None] * len(self.dataframe)
        q_gen_df_list = [None] * len(self.dataframe)
        for list_index, value in enumerate(machine_bus_values[0]):  # iterate over the machine values
            p_gen_df_list[machine_bus_nums[list_index] - 1] = value.real  # -1 is for the offset
            q_gen_df_list[machine_bus_nums[list_index] - 1] = value.imag

        self.dataframe['P Gen'] = p_gen_df_list
        self.dataframe['Q Gen'] = q_gen_df_list

    def addLoadinfo(self):
        """
        Descrtiption:
        input:
        output:
        """
        load_bus_nums = psspy.aloadint(-1, 4, "NUMBER")[1][0]  # get the bus numbers of buses with loads - list
        ierr, load_bus_values = psspy.aloadcplx(-1, 1, "MVAACT")  # load values
        if ierr != 0:
            raise Exception('Error in grabbing PGGEN values in addgen function')
        p_load_df_list = [None] * len(self.dataframe)
        q_load_df_list = [None] * len(self.dataframe)

        for list_index, value in enumerate(load_bus_values[0]):  # iterate over the machine values
            p_load_df_list[load_bus_nums[list_index] - 1] = value.real  # -1 is for the offset
            q_load_df_list[load_bus_nums[list_index] - 1] = value.imag
        self.dataframe['P Load'] = p_load_df_list
        self.dataframe['Q Load'] = q_load_df_list
