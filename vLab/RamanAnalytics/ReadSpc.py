import spc_spectra as spc
import pandas as pd
from os import listdir
from os.path import isfile, join


def read_spc(filename):
    """
    Read a single file
    :param filename: the spc file's name
    :return: loaded data
    """
    f = spc.File(filename)  # Read file
    # Extract X & y
    if f.dat_fmt.endswith('-xy'):
        for s in f.sub:
            x = s.x
            y = s.y
    else:
        for s in f.sub:
            x = f.x
            y = s.y

    Spec = pd.Series(y, index=x)

    return Spec


def read_spc_dir(directory, ext='.spc', orient='Row'):
    """
    #Read all files from directory and create a list, also ensures that the extension of file is correct
    :param ext:
    :param directory: the spc files' directory
    :param orient: the orientation of dataframe, column-wise or row-wise.
    :return: loaded data
    """

    flist = [f for f in listdir(directory) if isfile(join(directory, f)) and f.endswith(ext)]

    spectra_dict = {}
    # Read all of them an add them to a dictionary
    for file in flist:
        Spec = read_spc(directory + "/" + file)
        spectra_dict[file] = Spec

    # Decide the orientation of dataframe, column-wise or row-wise.
    if orient == 'Row':
        spectra_data_frame = pd.DataFrame(spectra_dict).transpose()
    else:
        spectra_data_frame = pd.DataFrame(spectra_dict)

    return spectra_data_frame, spectra_dict
