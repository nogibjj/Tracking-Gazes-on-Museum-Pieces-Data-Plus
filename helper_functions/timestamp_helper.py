"""Creating helper functions to manipulate the Unix Timestamps

Author : Eric Rios and Aditya John"""
import traceback
import pandas as pd
import numpy as np


def convert_timestamp_ns_to_ms(df, input_col_name="timestamp [ns]", output_col_name = 'heatmap_ts', subtract=True):
    """
    Simple function to convert the ns linux timestamp datetype to
    normal milliseconds of elapsed time
    """
    try:
        df[output_col_name] = pd.to_datetime(df[input_col_name])
        start_timestamp = df[output_col_name][0]
        if subtract:
            df[output_col_name] = df[output_col_name] - start_timestamp
        df[output_col_name] = df[output_col_name].astype(np.int64) / int(1e6)
        return df
    except:
        print(traceback.print_exc())
        return pd.DataFrame()
