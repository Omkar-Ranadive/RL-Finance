"""
Utility functions for processing data files
"""

from iex_parser import Parser, DEEP_1_0, TOPS_1_6
from constants import HIST_PATH
import pandas as pd
from datetime import datetime


def process_hist_data(file, hist_type, allowed_types, max_count=None):
    """
    Convert IEX HIST data to .csv files
    Args:
        file (str): String containing the file path of pcap file
        hist_type (str): Can be either TOPS or DEEP
        allowed_types (list) : List of message types to include in the processed .csv file
        Refer to Specifications on IEX site and IEX Parser documentation for more info:
        https://iextrading.com/trading/market-data/
        https://pypi.org/project/iex-parser/
        max_count (int): If specified, only max_count entries will be considered

    """

    # Set the parser based on hist type
    f_parser = DEEP_1_0 if hist_type == "DEEP" else TOPS_1_6
    df = pd.DataFrame()
    counter = 0

    with Parser(file, f_parser) as reader:
        for message in reader:
            if message['type'] in allowed_types:

                if message['type'] == 'price_level_update':
                    # Convert symbols and sides from byte-literals to strings
                    message['symbol'] = message['symbol'].decode('utf-8')
                    message['side'] = message['side'].decode('utf-8')

                if message['type'] == 'quote_update':
                    # Convert symbols from byte-literals to strings
                    message['symbol'] = message['symbol'].decode('utf-8')
                    # Ignore the "zero quote" cases
                    if message['bid_size'] == message['bid_price'] == message['ask_size'] == \
                            message['ask_price'] == 0:
                        continue

                df = df.append(message, ignore_index=True)
                counter += 1
                if counter % 1000 == 0:
                    print("Processed {} messages".format(counter))
                if max_count and counter > max_count:
                    break

    name = "processed_{}_{}.csv".format(hist_type, datetime.now().strftime("%d_%m_%Y_%H_%M"))
    df.to_csv(HIST_PATH / name, index=False)


if __name__ == '__main__':
    # file = str(HIST_PATH / 'data_feeds_20201124_20201124_IEXTP1_DEEP1.0.pcap.gz')
    # allowed = ['price_level_update']
    #
    # process_hist_data(file=file, allowed_types=allowed, hist_type="DEEP", max_count=30000)

    file = str(HIST_PATH / 'data_feeds_20201124_20201124_IEXTP1_TOPS1.6.pcap.gz')
    allowed = ['quote_update']

    process_hist_data(file=file, allowed_types=allowed, hist_type="TOPS", max_count=30000)
