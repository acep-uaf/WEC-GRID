# Run WEC-SIM Simulation for PSSe

def main():
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.cd("..\input_files\W2G_RM3")
    path = r'C:\Users\alex_barajas\Desktop\WEC-Sim\source' # Update to match your WEC-SIM source location
    eng.addpath(eng.genpath(path), nargout=0)
    eng.w2gSim(nargout=0)
    eng.WECsim_to_PSSe_dataFormatter(nargout=0)
    #print("sim complete")
    #eng.WECsim_to_PSSe_dataFormatter(nargout=0)
main()
