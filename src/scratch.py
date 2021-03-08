from iex_parser import Parser, DEEP_1_0, TOPS_1_6
from constants import HIST_PATH
#
# path_deep = r'W:\Finance_HIST\Feeds\data_feeds_20201124_20201124_IEXTP1_DEEP1.0.pcap.gz'
# path_tops = HIST_PATH / 'data_feeds_20201124_20201124_IEXTP1_TOPS1.6.pcap.gz'
# types = ['price_level_update', 'quote_update']
#
# counter = 0
#
#

# import numpy as np
# #
# with Parser(str(path_deep), DEEP_1_0) as reader:
#     for message in reader:
#         if message['type'] in types:
#
#             if message['type'] == 'price_level_update':
#                 # Convert symbols and sides from byte-literals to strings
#                 message['symbol'] = message['symbol'].decode('utf-8')
#                 message['side'] = message['side'].decode('utf-8')
#                 print(float(message['price']), np.round(float(message['price']), 2))
#
#             if message['type'] == 'quote_update':
#                 # Convert symbols from byte-literals to strings
#                 message['symbol'] = message['symbol'].decode('utf-8')
#                 # Ignore the "zero quote" cases
#                 if message['bid_size'] == message['bid_price'] == message['ask_size'] == \
#                         message['ask_price'] == 0:
#                     continue
#
#             counter += 1
#             if counter % 10000 == 0:
#                 print("Processed {} messages".format(counter))
#
from constants import DATA_PATH
from utils import load_file
import numpy as np
import timeit
allowed_types = ['price_level_update']
allowed_stocks_file = DATA_PATH / 'DEEP_2021_03_01-12_17_52_AM_T_100000_N_50'
allowed_stocks = load_file(filename=allowed_stocks_file)
file = str(HIST_PATH / 'data_feeds_20201124_20201124_IEXTP1_DEEP1.0.pcap.gz')


# process_hist_data(file=file, allowed_types=allowed, hist_type="DEEP", max_count=300000)
parser = Parser(file, DEEP_1_0)
reader = parser.__enter__()
reader_it = iter(reader)
counter = 0
message = next(reader_it)


while True:
    while message['type'] not in allowed_types or message['symbol'].decode('utf-8') not \
            in allowed_stocks:
        message = next(reader_it)

    # Convert symbols and sides from byte-literals to strings
    message['symbol'] = message['symbol'].decode('utf-8')
    message['side'] = message['side'].decode('utf-8')
    # Round off price to 2 decimal places
    message['price'] = np.round(float(message['price']), 2)
    counter += 1

    if counter % 100000:
        print("Counter: {}".format(counter))
