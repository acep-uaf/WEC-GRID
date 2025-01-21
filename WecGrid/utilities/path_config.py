from pathlib import Path
import os
import getpass


def find_folder_path(folder, start):
    directory = folder
    for root, dirs, files in os.walk(start):
        for name in dirs:
            if name == folder:
                return os.path.abspath(os.path.join(root, name))
    return None


def main():
    print("finding PSSe girl boss ... ")
    psse_path = find_folder_path("35.3", "C:\Program Files")
    print("Found! hehe :^)")
    print("okay now finding WEC-Sim... ")
    Wec_sim_path = find_folder_path("WEC-Sim", "C:\\Users\\" + getpass.getuser())
    print("it was right here lmao :^)")
    print("where's the WEC-Model ... ")
    Wec_model_path = find_folder_path("W2G_RM3", "C:\\Users\\" + getpass.getuser())
    print("there she is hehe :^)")

    if os.path.exists("path_config.txt"):
        os.remove("path_config.txt")
    else:
        print("The file does not exist")

    wec_grid_class = Path.cwd()
    wec_grid_folder = wec_grid_class.parent

    f = open("path_config.txt", "a")
    f.write(psse_path + "\n")
    f.write(Wec_sim_path + "\n")
    f.write(Wec_model_path + "\n")
    f.write(str(wec_grid_class) + "\n")
    f.write(str(wec_grid_folder))
    f.close()
    print("Paths written to path_config.txt")


main()
