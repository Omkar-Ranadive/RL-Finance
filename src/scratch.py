from iex_parser import Parser, DEEP_1_0, TOPS_1_6
from constants import HIST_PATH

path_deep = r'W:\Finance_HIST\Feeds\data_feeds_20201124_20201124_IEXTP1_DEEP1.0.pcap.gz'
path_tops = HIST_PATH / 'data_feeds_20201124_20201124_IEXTP1_TOPS1.6.pcap.gz'
types = ['price_level_update', 'quote_update']

counter = 0



import numpy as np
#
with Parser(str(path_deep), DEEP_1_0) as reader:
    for message in reader:
        if message['type'] in types:

            if message['type'] == 'price_level_update':
                # Convert symbols and sides from byte-literals to strings
                message['symbol'] = message['symbol'].decode('utf-8')
                message['side'] = message['side'].decode('utf-8')
                print(float(message['price']), np.round(float(message['price']), 2))

            if message['type'] == 'quote_update':
                # Convert symbols from byte-literals to strings
                message['symbol'] = message['symbol'].decode('utf-8')
                # Ignore the "zero quote" cases
                if message['bid_size'] == message['bid_price'] == message['ask_size'] == \
                        message['ask_price'] == 0:
                    continue

            counter += 1
            if counter % 1000 == 0:
                print("Processed {} messages".format(counter))

# for file in files:
#     reader = Parser(file, f_parser).__enter__()
#     cur_msg = iter(reader)
#     counter = 0
#     while counter < 10:
#         cur_msg = next(reader)
#         if cur_msg['type'] in allowed_types:
#             print(cur_msg)
#             counter += 1
