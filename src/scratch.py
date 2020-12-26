from iex_parser import Parser, DEEP_1_0, TOPS_1_6
from constants import HIST_PATH

path_deep = r'W:\LOB Data\data_feeds_20201124_20201124_IEXTP1_DEEP1.0.pcap.gz'
path_tops = HIST_PATH / 'data_feeds_20201124_20201124_IEXTP1_TOPS1.6.pcap.gz'
types = ['official_price', 'price_level_update']

counter = 0
# with Parser(path_deep, DEEP_1_0) as reader:
#     for message in reader:
#         if message['type'] == 'price_level_update':
#             print(message)
#             counter +=1
#             if counter > 100:
#                 break

#
# with Parser(str(path_tops), TOPS_1_6) as reader:
#     for message in reader:
#         if message['type'] == 'quote_update':
#             print(message)
#         # if message['type'] == 'price_level_update':
#         #     print(message)
#         #     counter +=1
#         #     if counter > 100:
#         #         break


# for file in files:
#     reader = Parser(file, f_parser).__enter__()
#     cur_msg = iter(reader)
#     counter = 0
#     while counter < 10:
#         cur_msg = next(reader)
#         if cur_msg['type'] in allowed_types:
#             print(cur_msg)
#             counter += 1
