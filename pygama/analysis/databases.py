import pandas as pd


def get_nonlinearity(db_path, channel):
    """ reads from a pandas hdf5 database """
    if 'db' not in get_nonlinearity.__dict__:
        get_nonlinearity.db = pd.read_hdf(db_path, key="data")

    return (get_nonlinearity.db.loc[channel].nonlin1,
            get_nonlinearity.db.loc[channel].nonlin2)
