"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

import argparse
import copy
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pandas as pd
import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm
import matplotlib.pyplot as plt

from nncodec.framework.applications.models.tokenizer import Tokenizer

import re
import pdb

DATA_CACHE_DIR = "data"

protocol_feat = ['direction', 'measured_qos', 'device', 'target_datarate']  # , 'scenario', 'drive_mode', 'operator']

cellular_conf_feat = ['PCell_E-ARFCN', 'PCell_Downlink_Num_RBs', 'PCell_Downlink_TB_Size', 'PCell_Downlink_RBs_MCS_0',
                      'PCell_Downlink_RBs_MCS_1', 'PCell_Downlink_RBs_MCS_2', 'PCell_Downlink_RBs_MCS_3',
                      'PCell_Downlink_RBs_MCS_4', 'PCell_Downlink_RBs_MCS_5', 'PCell_Downlink_RBs_MCS_6',
                      'PCell_Downlink_RBs_MCS_7', 'PCell_Downlink_RBs_MCS_8', 'PCell_Downlink_RBs_MCS_9',
                      'PCell_Downlink_RBs_MCS_10', 'PCell_Downlink_RBs_MCS_11', 'PCell_Downlink_RBs_MCS_12',
                      'PCell_Downlink_RBs_MCS_13', 'PCell_Downlink_RBs_MCS_14', 'PCell_Downlink_RBs_MCS_15',
                      'PCell_Downlink_RBs_MCS_16', 'PCell_Downlink_RBs_MCS_17', 'PCell_Downlink_RBs_MCS_18',
                      'PCell_Downlink_RBs_MCS_19', 'PCell_Downlink_RBs_MCS_20', 'PCell_Downlink_RBs_MCS_21',
                      'PCell_Downlink_RBs_MCS_22', 'PCell_Downlink_RBs_MCS_23', 'PCell_Downlink_RBs_MCS_24',
                      'PCell_Downlink_RBs_MCS_25', 'PCell_Downlink_RBs_MCS_26', 'PCell_Downlink_RBs_MCS_27',
                      'PCell_Downlink_RBs_MCS_28',
                      'PCell_Downlink_RBs_MCS_29', 'PCell_Downlink_RBs_MCS_30', 'PCell_Downlink_RBs_MCS_31',
                      'PCell_Downlink_Average_MCS', 'PCell_Uplink_Num_RBs', 'PCell_Uplink_TB_Size',
                      'PCell_Uplink_Tx_Power_(dBm)',
                      'PCell_Downlink_frequency', 'PCell_Uplink_frequency', 'PCell_Downlink_bandwidth_MHz',
                      'PCell_freq_MHz',
                      'PCell_Uplink_bandwidth_MHz', 'PCell_Band_Indicator',
                      # 'PCell_MCC', 'PCell_MNC_Digit', 'PCell_MNC', 'PCell_Allowed_Access',
                      'PCell_Cell_ID', 'PCell_Cell_Identity']  # , 'PCell_TAC']

cellular_conf_feat = ['PCell_E-ARFCN', 'PCell_Downlink_Num_RBs', 'PCell_Downlink_TB_Size',
                      'PCell_Uplink_Num_RBs', 'PCell_Uplink_TB_Size', 'PCell_Uplink_Tx_Power_(dBm)',
                      'PCell_Downlink_frequency', 'PCell_Uplink_frequency', 'PCell_freq_MHz',
                      'PCell_Downlink_bandwidth_MHz', 'PCell_Uplink_bandwidth_MHz', 'PCell_Band_Indicator',

                      'PCell_Downlink_RBs_MCS_0', 'PCell_Downlink_RBs_MCS_1', 'PCell_Downlink_RBs_MCS_2',
                      'PCell_Downlink_RBs_MCS_3', 'PCell_Downlink_RBs_MCS_4', 'PCell_Downlink_RBs_MCS_5',
                      'PCell_Downlink_RBs_MCS_6', 'PCell_Downlink_RBs_MCS_7', 'PCell_Downlink_RBs_MCS_8',
                      'PCell_Downlink_RBs_MCS_9', 'PCell_Downlink_RBs_MCS_10', 'PCell_Downlink_RBs_MCS_11',
                      'PCell_Downlink_RBs_MCS_12', 'PCell_Downlink_RBs_MCS_13', 'PCell_Downlink_RBs_MCS_14',
                      'PCell_Downlink_RBs_MCS_15', 'PCell_Downlink_RBs_MCS_16', 'PCell_Downlink_RBs_MCS_17',
                      'PCell_Downlink_RBs_MCS_18', 'PCell_Downlink_RBs_MCS_19', 'PCell_Downlink_RBs_MCS_20',
                      'PCell_Downlink_RBs_MCS_21', 'PCell_Downlink_RBs_MCS_22', 'PCell_Downlink_RBs_MCS_23',
                      'PCell_Downlink_RBs_MCS_24', 'PCell_Downlink_RBs_MCS_25', 'PCell_Downlink_RBs_MCS_26',
                      'PCell_Downlink_RBs_MCS_27', 'PCell_Downlink_RBs_MCS_28', 'PCell_Downlink_RBs_MCS_29',
                      'PCell_Downlink_RBs_MCS_30', 'PCell_Downlink_RBs_MCS_31', 'PCell_Downlink_Average_MCS',

                      # 'PCell_MCC', 'PCell_MNC_Digit', 'PCell_MNC', 'PCell_Allowed_Access',
                      'PCell_Cell_ID', 'PCell_Cell_Identity']  # , 'PCell_TAC']

gps_features = ['Latitude', 'Longitude', 'Altitude', 'speed_kmh', 'COG', 'Pos_in_Ref_Round']  # , 'area']

side_info_feat = ['precipIntensity', 'precipProbability', 'temperature', 'apparentTemperature', 'dewPoint',
                  'humidity', 'pressure', 'windSpeed', 'cloudCover', 'uvIndex', 'visibility', 'Traffic_Jam_Factor',
                  'Traffic_Street_Name', 'Traffic_Distance']

cellular_qos_feat = ['PCell_RSRP_1', 'PCell_RSRP_2', 'PCell_RSRP_max',
                     'PCell_RSRQ_1', 'PCell_RSRQ_2', 'PCell_RSRQ_max',
                     'PCell_RSSI_1', 'PCell_RSSI_2', 'PCell_RSSI_max',
                     'PCell_SNR_1', 'PCell_SNR_2',
                     'jitter', 'ping_ms', 'datarate']

cellular_loc_info = ['PCell_Cell_ID', 'PCell_Cell_Identity', 'PCell_TAC', 'area']

street_names = ['Altonaerstraße', 'Am', 'Schillertheater', 'An', 'der', 'Urania', 'Bachstraße', 'Behrenstraße',
                'Bismarckstraße', 'Brandenburgische', 'Straße', 'Corneliusbrücke', 'Ebertstraße', 'Ernst-Reuter-Platz', 'Funkturm',
                'Großer', 'Stern', 'Halenseestraße', 'Helmholtzstraße', 'Hohenzollerndamm', 'Invalidenstraße',
                'Joachimstaler', 'John-Foster-Dulles-Allee', 'Kaiser-Friedrich-Straße', 'Kaiserdamm',
                'Kaiserdamm-Süd', 'Kantstraße', 'Kemperplatz', 'Klingelhöfer', 'Knobelsdorffstraße',
                'Konstanzer', 'Kurfürstendamm', 'Kurfürstendamm(Schwarzbacher', 'Kurfürstenstraße',
                'Leibnizstraße', 'Lennèstraße', 'Lietzenburger', 'Marchstraße', 'Masurenallee', 'Messedamm',
                'Nürnberger', 'Olivaer', 'Olof-Palme-Platz', 'Otto-Suhr-Allee', 'Paulsborner',
                'Platz', 'des', 'März', 'Rankeplatz', 'Rathenauplatz', 'Richard-Wagner-Straße', 'Rudolstädter',
                'Seesener', 'Sophie-Charlotten-Straße', 'Spichernstraße', 'Spreeweg', 'Juni',
                'Suarezstraße', 'Süd', 'Tauentzienstraße', 'Tiergartenstraße', 'Uhlandstraße', 'Westfälische',
                'Württembergische', 'Xantener', 'Yitzhak-Rabin-Straße', 'Straße)', '17.', '18.']

"""
These message types are logs that mobile diagnostic tools (such as MobileInsight) use to monitor cellular network activities, e
specially for LTE networks. Each message type provides different types of insights into the network's state, performance, 
and user equipment (UE) interactions. Here’s a breakdown:

LTE_PHY_Serv_Cell_Measurement: 
This message reports on physical layer measurements for the serving cell in LTE. It 
includes metrics like the Reference Signal Received Power (RSRP), Reference Signal Received Quality (RSRQ), and 
Signal-to-Interference-plus-Noise Ratio (SINR). These metrics help assess signal strength and quality, which are essential for maintaining a stable connection.

LTE_RRC_Serv_Cell_Info: 
This message provides information on the serving cell at the Radio Resource Control (RRC) layer. 
It includes details such as the cell ID, tracking area code, and parameters related to the RRC state of the UE (e.g., 
whether it’s in an idle or connected mode). This message is crucial for understanding the control plane’s management of the UE's connection to the cell.

LTE_PHY_PDSCH_Stat_Indication: 
The Physical Downlink Shared Channel (PDSCH) is the main downlink data channel in LTE, 
and this message indicates the status of PDSCH receptions. It includes data on metrics like block error rate (BLER), 
signal strength, and decoding success rates. This information is useful for understanding downlink performance and the 
quality of data transmission from the network to the UE.

LTE_PHY_PUSCH_Tx_Report: 
This message reports on transmissions from the UE on the Physical Uplink Shared Channel (PUSCH), 
the primary uplink data channel in LTE. It includes details on transmission power, the amount of data sent, and 
modulation/coding schemes used. This data is vital for analyzing uplink performance and the efficiency of data transmission from the UE to the network.


### Configuration Features
These features are mostly identifiers, static configuration values, or structural parameters, which define the setup 
of the network but are not direct indicators of network quality. They can guide or be adjusted based on feedback but 
don't themselves serve as feedback for quality measurement.

- **device**: Identifier of the device, not a direct quality metric.
- **PCell_E-ARFCN**: Absolute Radio Frequency Channel Number; this is a static frequency configuration.
- **PCell_Downlink_Num_RBs**: Number of downlink Resource Blocks allocated, part of network configuration.
- **PCell_Uplink_Num_RBs**: Number of uplink Resource Blocks, part of network configuration.
- **PCell_Downlink_TB_Size**: Transport Block Size for downlink, related to throughput configuration.
- **PCell_Uplink_TB_Size**: Transport Block Size for uplink, related to throughput configuration.
- **PCell_Downlink_RBs_MCS_0 to PCell_Downlink_RBs_MCS_31**: Modulation and Coding Scheme (MCS) indices for different Resource Blocks, related to encoding configuration for transmission.
- **PCell_Downlink_Average_MCS**: Average MCS for downlink, tied to encoding and modulation configuration.
- **PCell_Uplink_Tx_Power_(dBm)**: Transmission power for uplink, a configuration for signal strength.
- **PCell_Cell_ID**: Cell identifier; used for network and cell identification.
- **PCell_Downlink_frequency**: Downlink frequency configuration.
- **PCell_Uplink_frequency**: Uplink frequency configuration.
- **PCell_Downlink_bandwidth_MHz**: Downlink bandwidth, part of the cell’s setup.
- **PCell_Uplink_bandwidth_MHz**: Uplink bandwidth, part of the cell’s setup.
- **PCell_Cell_Identity**: Unique cell identifier in the network.
- **PCell_TAC**: Tracking Area Code, an identifier for location tracking.
- **PCell_Band_Indicator**: Band indicator, part of the frequency and band configuration.
- **PCell_MCC**: Mobile Country Code, identifying the country.
- **PCell_MNC_Digit**: Mobile Network Code digit, part of operator identification.
- **PCell_MNC**: Full Mobile Network Code, identifying the operator.
- **PCell_Allowed_Access**: Indicates the type of allowed access; configuration for network access control.
- **PCell_freq_MHz**: Frequency in MHz, which is a configuration feature.

### Quality Features (for Feedback and Optimization)
These features are indicators of network performance or quality metrics that provide feedback on the network's 
effectiveness, user experience, or signal strength. They can be monitored and used to optimize network configurations 
to improve connectivity and performance.

- **ping_ms**: Ping time in milliseconds, a direct measure of network latency and quality.
- **datarate**: Data rate or throughput, indicating connection speed and quality.
- **jitter**: Variation in latency, a quality metric reflecting stability.
- **PCell_RSRP_1, PCell_RSRP_2, PCell_RSRP_max**: Reference Signal Received Power; indicates signal strength quality.
- **PCell_RSRQ_1, PCell_RSRQ_2, PCell_RSRQ_max**: Reference Signal Received Quality; indicates signal-to-noise ratio and interference.
- **PCell_RSSI_1, PCell_RSSI_2, PCell_RSSI_max**: Received Signal Strength Indicator; indicates the power of the received signal.
- **PCell_SNR_1, PCell_SNR_2**: Signal-to-Noise Ratio, a measure of signal quality.

In summary:
- **Configuration Features**: These are the settings or identifiers defining network setup and structure.
- **Quality Features**: These are measurements that reflect the actual performance or quality of the network, providing feedback that can guide optimization efforts.
"""


# def get_feat_string_from_row(row):
#     str_protocol_feat = ' '.join([f'{key} = {row[key]:.3f}' if isinstance(row[key], float) and pd.notna(
#         row[key]) else f'{key} = {row[key] if pd.notna(row[key]) else ""}' for key in protocol_feat])
#     str_cellular_conf_feat = ' '.join([f'{key} = {row[key]:.3f}' if isinstance(row[key], float) and pd.notna(
#         row[key]) else f'{key} = {row[key] if pd.notna(row[key]) else ""}' for key in cellular_conf_feat])
#     str_gps_features = ' '.join([f'{key} = {row[key]:.3f}' if isinstance(row[key], float) and pd.notna(
#         row[key]) else f'{key} = {row[key] if pd.notna(row[key]) else ""}' for key in gps_features])
#     str_side_info_feat = ' '.join([f'{key} = {row[key]:.3f}' if isinstance(row[key], float) and pd.notna(
#         row[key]) else f'{key} = {row[key] if pd.notna(row[key]) else ""}' for key in side_info_feat])
#     str_cellular_qos_feat = ' '.join([f'{key} = {row[key]:.3f}' if isinstance(row[key], float) and pd.notna(
#         row[key]) else f'{key} = {row[key] if pd.notna(row[key]) else ""}' for key in cellular_qos_feat])
#     seq_write = ' --> '.join(
#         [str_protocol_feat, str_cellular_conf_feat, str_gps_features, str_side_info_feat, str_cellular_qos_feat])
#     return seq_write

# def format_with_spaces(value):
#    # Check if the value is a float, integer, or a digit string
#    if isinstance(value, float):
#        formatted = f"{value:.3f}"  # Format floats with three decimal places
#    elif isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
#        formatted = str(value)  # Convert integer or digit string to plain string
#    else:
#        return value  # Return the value as-is if it's not numeric
#
#    # Insert spaces between each character
#    return ' '.join(formatted)


def format_with_spaces(value):
    # Check if the value is a float, integer, or a digit string
    if isinstance(value, float):
        return value
    elif isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        return value
    else:
        return value  # Return the value as-is if it's not numeric
    # Insert spaces between each character
    return ' '.join(formatted)

# def get_feat_string_from_row(row):
#     def format_value(key):
#         value = row[key]
#         if pd.notna(value):  # Check if the value is not NaN
#             val = format_with_spaces(value)
#             if isinstance(val, float):
#                 val = f"{val:.3g}"
#                 # if "e" in val: # Expand scientific notation
#                 #     val = f"{val:.{abs(int(val)):d}f}".rstrip("0").rstrip(".")
#                 if "e" in val: # Expand scientific notation
#                     coeff, exp = val.split("e")
#                     exp = int(exp)
#                     coeff = float(coeff)
#                     result = coeff * (10 ** exp)
#                     decimal_places = max(0, -exp + 1)
#                     val = f"{result:.{decimal_places}f}".rstrip("0").rstrip(".")
#             return f"{key} = {val}"
#         else:
#             return f"{key} = "

def get_feat_string_from_row(row):
    def format_value(key):
        value = row[key]
        if pd.notna(value):  # Check if the value is not NaN
            val = format_with_spaces(value)
            if isinstance(val, float):
                if abs(val) >= 1:
                    # For numbers >= 1: 2 decimals max
                    val = f"{val:.3f}".rstrip("0").rstrip(".")
                else:
                    # For numbers < 1: keep 2-3 significant digits
                    digits = 3
                    abs_val = abs(val)
                    while abs_val < 0.1 and digits < 5:  # Increase precision if number is very small
                        digits += 1
                        abs_val *= 10
                    val = f"{val:.{digits}f}".rstrip("0").rstrip(".")
            return f"{key} = {val}"
        else:
            return f"{key} = "
    str_protocol_feat = ' '.join([format_value(key) for key in protocol_feat])
    str_cellular_conf_feat = ' '.join([format_value(key) for key in cellular_conf_feat])
    str_gps_features = ' '.join([format_value(key) for key in gps_features])
    str_side_info_feat = ' '.join([format_value(key) for key in side_info_feat])
    str_cellular_qos_feat = ' '.join([format_value(key) for key in cellular_qos_feat])

    seq_write = ' --> '.join(
        [str_protocol_feat, str_cellular_conf_feat, str_gps_features, str_side_info_feat, str_cellular_qos_feat]
    )
    return seq_write




def train_telko_vocab(max_sentences=1e9, data_path='', out_dir='out', split='train'):
    """
    Trains a custom sentencepiece tokenizer on the BerlinV2X dataset
    """

    print("Will now train the vocab...")

    # all_files = sorted(glob.glob(os.path.join(data_path, "*.uint16")))
    data_set = os.path.join(data_path, f"cellular_dataframe.parquet")
    df = pd.read_parquet(data_set)

    # df_side = pd.read_parquet(os.path.join(data_path, f"sidelink_dataframe.parquet"))


    # data_setsource = os.path.join(data_path, f"sources/mobile_insight/pc1/LTE_RRC_Serv_Cell_Info.parquet")
    # df_source_pc1 = pd.read_parquet(data_setsource)



    # df = df.query("`area`!= 'UNKNOWN'")
    # df[datarate_label] = df['datarate'] / 1e6
    df = df.query("`operator`== 1")  # Telekom only
    df.columns = df.columns.str.replace(' ', '_')

    def unique_vals_cts_df_kef(df, key):
        temp_df = df[df[key].notna()]
        df_np = temp_df[key].to_numpy()
        return np.unique(df_np, return_counts=True)

    # df = df[df['area'] == "Avenue"]

    uqe_cellID = unique_vals_cts_df_kef(df, 'PCell_Cell_ID')
    uqe_cellIdent = unique_vals_cts_df_kef(df, 'PCell_Cell_Identity')
    uqe_cellTAC = unique_vals_cts_df_kef(df, 'PCell_TAC')
    uqe_cellMCC = unique_vals_cts_df_kef(df, 'PCell_MCC')  # unary --> redundant info for Telko operator
    uqe_cellMNCD = unique_vals_cts_df_kef(df, 'PCell_MNC_Digit')  # unary --> redundant info for Telko operator
    uqe_cellMNC = unique_vals_cts_df_kef(df, 'PCell_MNC')  # unary --> redundant info for Telko operator

    uqe_cellDlF = unique_vals_cts_df_kef(df, 'PCell_Downlink_frequency')
    uqe_cellUlF = unique_vals_cts_df_kef(df, 'PCell_Uplink_frequency')
    uqe_cellDlB = unique_vals_cts_df_kef(df, 'PCell_Downlink_bandwidth_MHz')
    uqe_cellUlB = unique_vals_cts_df_kef(df, 'PCell_Uplink_bandwidth_MHz')
    uqe_cellAA = unique_vals_cts_df_kef(df, 'PCell_Allowed_Access')  # unary --> redundant info for Telko operator
    uqe_cellf = unique_vals_cts_df_kef(df, 'PCell_freq_MHz')

    uqe_cellARFCN = unique_vals_cts_df_kef(df, 'PCell_E-ARFCN')
    uqe_cellNRBs = unique_vals_cts_df_kef(df, 'PCell_Downlink_Num_RBs')
    uqe_cellTBS = unique_vals_cts_df_kef(df, 'PCell_Downlink_TB_Size')

    uqe_cellscene= unique_vals_cts_df_kef(df, 'scenario')
    uqe_celldrmode = unique_vals_cts_df_kef(df, 'drive_mode')
    uqe_celltrate= unique_vals_cts_df_kef(df, 'target_datarate')
    uqe_cellmQoS = unique_vals_cts_df_kef(df, 'measured_qos')
    uqe_celldir = unique_vals_cts_df_kef(df, 'direction')
    uqe_cellstreetnames = unique_vals_cts_df_kef(df, 'Traffic_Street_Name')


    # cell_id_percentile = np.percentile(unique_cell_ids_count, 99)
    # cell_id_max = np.max(unique_cell_ids_count)
    # min_num_cell_data = cell_id_max-cell_id_percentile

    # cell_ids_freqused = uqe_cellID[0][uqe_cellID[1] >= 2000]


    # df = df[df['PCell_Cell_ID'] == 2]

    # cell_ids = cell_ids.unique()

    # print(f"Number of unique PCell_Cell_ID values: {unique_cell_ids_count}")
    #
    # # Plot a histogram of the PCell_Cell_ID
    # plt.figure(figsize=(10, 6))
    # cell_ids.hist(bins=240)
    # plt.title('Histogram of PCell_Cell_ID')
    # plt.xlabel('PCell_Cell_ID')
    # plt.ylabel('Frequency')
    # plt.show()

    txt_dir = os.path.join(data_path, f"V2X_to_text")
    os.makedirs(txt_dir, exist_ok=True)

    tokenized_filename = os.path.join(txt_dir, f"tokenizer_train_data.txt")

    features = df.columns.values.tolist()

    sampled_df = df.sample(frac=0.03, random_state=909) # for toxenizer training use only a portion  of 3%
    with open(tokenized_filename, "w") as f:
        for time, row in sampled_df.iterrows():
            txt_seq = get_feat_string_from_row(row)
            daytime = f"time = {' '.join(time.strftime('%H%M'))} "
            txt_seq = daytime + txt_seq
            f.write(txt_seq + '\n')
    f.close()

    result_tokens = ['uplink', 'downlink', 'delay', 'pc1', 'pc4'] #, 'Residential', 'Park', 'Avenue', 'Highway', 'Tunnel']

    possible_tokens = ','.join([f'{str(ai)}' for ai in range(0, 10)] + [".", "-->", "-", "=", "time"] +
                                list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß'] +
                                protocol_feat + cellular_conf_feat + gps_features + side_info_feat + cellular_qos_feat +
                                street_names + result_tokens)


    print(possible_tokens.count(","))
    # vocsize = possible_tokens.count(",") + 1 + 3 # +3 for UNK, BOS, EOS
    # vocsize = possible_tokens.count(",") + 1 + 1 # +1 for UNK (if EOS and BOS disabled)

    # tokenized_filename="/Users/adm_becking/PycharmProjects/data/V2X_dataset/V2X_to_text/tokenizer_train_data.txt"
    spm.SentencePieceTrainer.Train(input=tokenized_filename,
                                   # add_dummy_prefix="false",
                                   # model_type="word", #"word",
                                   model_prefix=f"{out_dir}/telko_tokenizer",
                                   user_defined_symbols=possible_tokens,
                                   # control_symbols=possible_tokens,
                                   # required_chars=possible_tokens,
                                   vocab_size=237,
                                   # split_by_number="true",
                                   # max_sentencepiece_length=4,
                                   character_coverage=1.0,
                                   # use_all_vocab="true",
                                   # pad_piece="",
                                   # add_dummy_prefix="false",
                                   remove_extra_whitespaces="true",
                                   split_by_unicode_script="false",
                                   treat_whitespace_as_suffix="false",
                                   allow_whitespace_only_pieces="false",
                                   # bos_id=-1,
                                   # eos_id=-1
                                   )

    print(f"Trained tokenizer is in {out_dir}/telko_tokenizer.model")

    enc = Tokenizer(tokenizer_model=f"{out_dir}/telko_tokenizer.model")
    vocab_size = enc.sp_model.get_piece_size()
    vocabulary = [enc.sp_model.id_to_piece(i) for i in range(vocab_size)]


    print("Done.")

def process_shard(args, vocab_size):
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    with open(shard, "r") as f:
        data = json.load(f)
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
        all_tokens.extend(tokens)
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # calculate the output filename
    if vocab_size == 0:
        # if we're using Llama 2, just save the tokenized file in the same dir
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        # save .bin files into a new tok{N} directory
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def process_shard_txt(args, vocab_size):
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    with open(shard, "r") as f:
        data = f.read() #json.load(f)
    all_tokens = []
    # for example in tqdm(data, position=shard_id):
    #     # text = example["story"]
    text = data.strip()  # get rid of leading/trailing whitespace
    tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
    all_tokens.extend(tokens)
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # calculate the output filename
    if vocab_size == 0:
        # if we're using Llama 2, just save the tokenized file in the same dir
        tokenized_filename = shard.replace(".txt", ".bin")
    else:
        # save .bin files into a new tok{N} directory
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".txt", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def process_shard_json_framewise(args, vocab_size, max_seq_len=257):
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    with open(shard, "r") as f:
        data = json.load(f)

    for fidx, frame in enumerate(data):
        if frame["SliceType"] != 2: ## only intra frames
            continue
        all_tokens = []
        for ctx_id in frame:
            if not ctx_id.isdigit():
                continue
            bin_seq = frame[ctx_id][3]
            if len(bin_seq) < max_seq_len:
                continue
            text = bin_seq.strip()  # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
            all_tokens.extend(tokens)

        # convert to uint16 nparray
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        # calculate the output filename
        if vocab_size == 0:
            # if we're using Llama 2, just save the tokenized file in the same dir
            tokenized_filename = shard.replace(".json", ".bin")
        else:
            # save .bin files into a new tok{N} directory
            bin_dir = os.path.join(DATA_CACHE_DIR, f"ECM/train/tok{vocab_size}")
            shard_basename = os.path.basename(shard)
            bin_basename = shard_basename.replace(".json", ".bin")
            tokenized_filename = os.path.join(bin_dir, f"frame{fidx}_{bin_basename}")
        # write the bytes
        with open(tokenized_filename, "wb") as f:
            f.write(all_tokens.tobytes())
        # calculate the average sequence length (they are separated by BOS=1)
        avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
        print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")

def process_shard_json(args, vocab_size, ctxmdl_id=1, split='train', dest_dir=DATA_CACHE_DIR, out_dir=""):
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size, ctxmdl_id=ctxmdl_id, out_dir=out_dir)
    enc = Tokenizer(tokenizer_model)
    with open(shard, "r") as f:
        data = json.load(f)

    for fidx, frame in enumerate(data):
        if frame["SliceType"] != 2: ## only intra frames
            continue

        if ctxmdl_id is not None:
            bin_seq = frame[ctxmdl_id][3]
            text = bin_seq.strip()  # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=False, eos=False)  # encode the text, use BOS
            # convert to uint16 nparray
            all_tokens = np.array(tokens, dtype=np.uint16)
            # save .bin files into a new tok{N} directory
            if dest_dir == DATA_CACHE_DIR:
                bin_dir = os.path.join(DATA_CACHE_DIR, f"ECM/{split}/tok{vocab_size}")
            else:
                bin_dir = dest_dir
            tokenized_filename = os.path.join(bin_dir, f"CTX{ctxmdl_id}.bin")
            # append the bytes
            with open(tokenized_filename, "ab") as f:
                f.write(all_tokens.tobytes())

        else:
            for ctx_id in frame:
                if not ctx_id.isdigit():
                    continue
                bin_seq = frame[ctx_id][3]
                text = bin_seq.strip()  # get rid of leading/trailing whitespace
                tokens = enc.encode(text, bos=False, eos=False)  # encode the text, use BOS
                # convert to uint16 nparray
                all_tokens = np.array(tokens, dtype=np.uint16)
                # save .bin files into a new tok{N} directory
                if dest_dir == DATA_CACHE_DIR:
                    bin_dir = os.path.join(DATA_CACHE_DIR, f"ECM/{split}/tok{vocab_size}")
                else:
                    bin_dir = dest_dir
                tokenized_filename = os.path.join(bin_dir, f"CTX{ctx_id}.bin")
                # append the bytes
                with open(tokenized_filename, "ab") as f:
                    f.write(all_tokens.tobytes())

def process_shard_uint16(args, dest_dir=DATA_CACHE_DIR, out_dir="", chunk_size=int(1e6)):
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(ctxmdl_id="all", out_dir=out_dir)

    enc = Tokenizer(tokenizer_model)

    # vocab_size = enc.sp_model.get_piece_size()
    # vocabulary = [enc.sp_model.id_to_piece(i) for i in range(vocab_size)]
    #
    # tokens = enc.encode("0 5 1 5 0 5 1 9 1 9 0 10", bos=False, eos=False)

    with open(shard, "r") as f:

        data = np.fromfile(shard, dtype=np.uint16)

        if data.size > chunk_size:
            num_batches = (data.size // chunk_size) + 1
        else:
            num_batches = 1

        for i in range(num_batches):
            start = i * chunk_size
            end = start + chunk_size

            chunk = data[start:end]

            # ctx_idx = (np.right_shift(chunk, 1) + 2).astype(str)
            ctx_idx = (np.right_shift(chunk, 1)).astype(str)
            bins = np.bitwise_and(chunk, 1).astype(str)
            # seq_str = ' '.join([f'{ai} {bi}' for ai, bi in zip(bins, ctx_idx)])
            seq_str = ' '.join([f'{ai}={bi}' for ai, bi in zip(bins, ctx_idx)])
            # seq_str = ' '.join([f'{ai}={bi}' for ai, bi in zip(bins, ctx_idx) if bi != '1024']) # for excluding EP bins

            text = seq_str.strip()  # get rid of leading/trailing whitespace
            # del chunk, ctx_idx, bins, seq_str

            tokens = enc.encode(text, bos=False, eos=False)  # encode the text, use BOS
            # convert to uint16 nparray
            all_tokens = np.array(tokens, dtype=np.uint16)
            # del tokens
            name = shard.split("/")[-1].replace(".uint16", "")
            tokenized_filename = os.path.join(dest_dir, f"{name}.bin")
            # append the bytes
            with open(tokenized_filename, "ab") as f:
                f.write(all_tokens.tobytes())


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def process_parquet(args, dest_dir=DATA_CACHE_DIR, out_dir="", mixed_token=False):
    shard_id, df = args

    # tokenizerpath = f"{out_dir}/telko_tokenizer.model"
    tokenizerpath = "../tokenizer/telko_tokenizer.model"
    enc = Tokenizer(tokenizerpath)

    vocab_size = enc.sp_model.get_piece_size()
    vocabulary = [enc.sp_model.id_to_piece(i) for i in range(vocab_size)]
    # tokens = enc.encode("0 5 1 5 0 5 1 9 1 9 0 10", bos=False, eos=False)
    # tokenized_filename = os.path.join(dest_dir, f"clientID_{shard_id}.bin")


    number_of_tokens_per_sample = []
    all_samples = []
    # with open(tokenized_filename, "ab") as f:
    for time, row in df.iterrows():
        txt_seq = get_feat_string_from_row(row).strip()
        # print(txt_seq)
        # daytime = f"time = {' '.join(time.strftime('%H%M'))} "
        daytime = f"time = {time.hour+time.minute/60} "
        txt_seq = daytime + txt_seq

        str_token_list = [word if not is_number(word) else "*" for word in txt_seq.split() ]
        number_token_list = [number  for number in txt_seq.split() if is_number(number)]
        #for word in str_token_list:
        #    if enc.sp_model.piece_to_id(word) == 0:
        #        print(f"Unknown token: {word}")

        # for word in txt_seq.split():
        #     print(enc.sp_model.piece_to_id(word))
        #
        # for piece in txt_seq.split():
        #     print(piece, enc.sp_model.piece_to_id(piece))
        #
        # txt_seq = txt_seq.replace("=", " = ").replace("-->", " --> ")
        #
        # txt_seq = " ".join(txt_seq.split())
        # pdb.set_trace()
        tokens = enc.encode(' '.join(str_token_list), bos=False, eos=False)

        # contains_nans = (np.array(tokens)<0).any()
        # if contains_nans:
        #     t=0



        # tokens = enc.encode(txt_seq, bos=False, eos=False)
        number_token_list = np.array(number_token_list, dtype=float)
        #print(f'Min: {number_token_list.min()}, max: {number_token_list.max()}, mean: {number_token_list.mean()}')
        # normalize token list

        iter_a = iter(number_token_list)



        if not mixed_token:# To evaluate the MOE-Transformers performance against intrepreting the numbers as word tokens, The following line has to be set to "True"
            combined = [[0, tok] for tok in enc.encode(txt_seq, bos=False, eos=False)]
        else:
            # word tokens are numbers and end up as (0, token)
            # numerical tokens are 0 end end up as (1, number)
            combined = [[0, tok] if tok != 0 else [1, next(iter_a)] for tok in tokens]

        all_tokens = np.array(combined, dtype=np.float32)
        number_of_tokens_per_sample.append(all_tokens.shape[0])
        # np.save(f, all_tokens)
        # pdb.set_trace()
        #all_tokens = np.array(tokens, dtype=np.uint8)
        #f.write(all_tokens.tobytes())
        all_samples.append(all_tokens)


    np.savez(os.path.join(dest_dir, f"clientID_{shard_id}.npz"), *all_samples)

    print(f"max sequence length of samples: {np.max(number_of_tokens_per_sample)}")
    print(f"number of samples: {len(number_of_tokens_per_sample)}")
    print(f"number of tokens:: {np.sum(number_of_tokens_per_sample)}")


def pretokenize(vocab_size):
    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if vocab_size > 0:
        # .bin files will be saved into tok{N} directory, create it once here
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    # process all the shards in a process pool
    fun = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    print("Done.")


def pretokenize_telko(data_path, split='train', out_dir="", normalization=False):
    # iterate the shards and tokenize all of them one by one

    data_dir = os.path.join(data_path, f"{split}")

    print(f"write tokenized bin sequences to binary files in {data_dir}")
    os.makedirs(data_dir, exist_ok=True)

    data_set = os.path.join(data_path, f"cellular_dataframe.parquet")
    df = pd.read_parquet(data_set)
    # pdb.set_trace()
    df = df.query("`operator`== 1")  # Telekom only
    df.columns = df.columns.str.replace(' ', '_')

    #delete the following columns because they are empty (at least for `operator`== 1)
    df = df.drop("SCell_Uplink_Num_RBs", axis=1)
    df = df.drop("SCell_Uplink_TB_Size", axis=1)
    df = df.drop("SCell_Uplink_Tx_Power_(dBm)", axis=1)


    split_df = df.sample(frac=0.90, random_state=909)

    ## normalization
    if normalization:
        mean_frame = split_df.select_dtypes(include='number').mean(axis=0)
        std_frame = split_df.select_dtypes(include='number').std(axis=0)


        df_standard_numeric = (df.select_dtypes(include='number') - mean_frame) / (std_frame + 1e-10)
        df.loc[:,df_standard_numeric.columns] = df_standard_numeric[df_standard_numeric.columns]

        # Save the dictionary to a JSON file for denormalization demonstrator
        mean_dict, std_dict = mean_frame.to_dict(), std_frame.to_dict()
        with open('mean_dict.json', 'w') as meanjson_file:
            json.dump(mean_dict, meanjson_file)
        with open('std_dict.json', 'w') as stdjson_file:
            json.dump(std_dict, stdjson_file)


        ### outlier detection & removal:
        numeric_cols = df.select_dtypes(include='number')
        numeric_cols_filled = numeric_cols.fillna(0) # temporary fill NaNs with 0 to not affect filtering
        df = df[(numeric_cols_filled.abs() <= 6).all(axis=1)] # Remove samples where any numeric column has an absolute value >= 6


        split_df = df.sample(frac=0.90, random_state=909)


    if split == 'train':
        tok_df = split_df

        client_identifier_key = 'area'  # 'PCell_TAC'
        client_identifier = ['Residential', 'Park', 'Avenue', 'Highway', 'Tunnel']  # [1492, 1493, 1494, 1495]

        shard_filenames = [tok_df[tok_df[client_identifier_key] == client_identifier[client]] for client in
                           range(len(client_identifier))]

        for client in range(len(client_identifier)):
            process_parquet((client, shard_filenames[client]), dest_dir=data_dir, out_dir=out_dir, mixed_token=normalization)

    elif split == 'test':
        tok_df = df.drop(split_df.index)
        shard_filenames = [tok_df]

        process_parquet((0, shard_filenames[0]), dest_dir=data_dir, out_dir=out_dir, mixed_token=normalization)

    # pdb.set_trace()# the following only for debugging
    # process_parquet((0,shard_filenames[0]), dest_dir=data_dir, out_dir=out_dir)



    # fun = partial(process_parquet, dest_dir=data_dir, out_dir=out_dir)
    # with ProcessPoolExecutor() as executor:
    #     executor.map(fun, enumerate(shard_filenames))
    print("Done.")



def number_of_bins(data_dir=".", split="train", ctxmdl_id=None):
    data_dirsplit = os.path.join(data_dir, f"{split}")
    shard_filenames = sorted(glob.glob(os.path.join(data_dirsplit, "*.json")))
    num_of_bins_per_video = {}
    for shard in shard_filenames:

        with open(shard, "r") as f:
            data = json.load(f)

        num_bins_frames = 0
        num_frames = 0
        for frame in data:
            if isinstance(frame, dict):
                if frame["SliceType"] != 2: ## only intra frames
                    continue
                for ctx_id in frame:
                    if not ctx_id.isdigit():
                        continue
                    if ctxmdl_id is not None and ctx_id != ctxmdl_id:
                        continue
                    num_bins_frames += len(frame[ctx_id][3])
                    num_frames += 1
        num_of_bins_per_video[shard.split("/")[-1]] = (num_bins_frames, num_frames)

    def sort_by_num_bins(item):
        return item[1][0]
    num_of_bins_per_video = dict(sorted(num_of_bins_per_video.items(), key=sort_by_num_bins, reverse=True))

    total_bins = 0
    for vid in num_of_bins_per_video:
        bins, frames = num_of_bins_per_video[vid]
        total_bins += bins
        if split == "test":
            print(f"{bins} bins in {frames} frames for sequence '{vid}'")

    print(f"total {split} bins of ctxmdl {ctxmdl_id}: {total_bins}")
    return total_bins


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        if self.vocab_source == "llama2":
            # the .bin files are right along the .json files
            bin_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        elif self.vocab_source == "custom":
            # the .bin files are in tok{N} directory
            bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        assert len(shard_filenames)>0, f"No bin files found in {bin_dir}"
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                if num_batches < 1:
                    print("NUM_BATCHES < 1!!!")
                    continue
                # assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


class PretokDatasetuint16(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_source, data_dir):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        # self.vocab_size = vocab_size
        self.vocab_source = vocab_source
        # self.use_ctxmdl_prefix = use_ctxmdl_prefix
        self.bin_dir = data_dir
        # self.ctxmdl_id = ctxmdl_id

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        # the .bin files are in tok{N} directory
        # if self.bin_dir == None:
        #     bin_dir = os.path.join(DATA_CACHE_DIR, f"ECM/{self.split}/tok{self.vocab_size}")
        # else:
        bin_dir = os.path.join(self.bin_dir, f"{self.split}/{self.split}_preprocessed")

        shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))

        # train/test split. let's use only shard 0 for test split, rest train
        # shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]

        # trainsplit = int(np.floor(len(shard_filenames) * 0.9))
        # shard_filenames = shard_filenames[:trainsplit] if self.split == "train" else shard_filenames[trainsplit:]
        # shard_filenames = [s for s in shard_filenames if f"CTX{self.ctxmdl_id}.bin" in s]

        assert len(shard_filenames)>0, f"No bin files found in {bin_dir}"

        num_strides = 8

        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                # generates a binary code for ctx mdl index of length 10:
                # if self.use_ctxmdl_prefix:
                #     bin_ctx_id = format(int(''.join(filter(str.isdigit, shard.split("/")[-1]))), '010b')
                #     bin_ctx_id = bin_ctx_id.replace('1', '3')
                #     bin_ctx_id = bin_ctx_id.replace('0', '4')
                #     bin_ctx_id_tensor = torch.tensor([int(d) for d in bin_ctx_id])

                m = np.memmap(shard, dtype=np.uint16, mode="r")

                # print(m[:100])

                num_batches = len(m) // self.max_seq_len
                num_batches -= 2  # drop the last partial batch minus the last full batch for sliding operation
                if num_batches < 1:
                    continue
                # assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)

                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    for s in range(num_strides):
                        chunk = torch.from_numpy((m[start+s:end+s]).astype(np.int64))

                        # if chunk[-1] not in [3, 4]:
                        #     continue

                        # if (chunk == 1).any(): # checks if chunk includes a BOS (i.e., new ctx mdl) and avoids this chunk if so
                        #     continue
                        if (chunk == 0).any(): # checks if chunk includes a <unk> (i.e., new ctx mdl) and avoids this chunk if so
                            print("WARNING: <unk> token identified, chunk skipped")
                            continue
                        # if (chunk == 2).any(): # checks if chunk includes a EOS (i.e., new ctx mdl) and avoids this chunk if so
                        #     continue

                        x = chunk[:-1]
                        y = chunk[1:]
                        # y = chunk[-1]
                        yield x, y

class PretokDatasetuint16test(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, tok_sequence, max_seq_len):
        super().__init__()
        self.tok_sequence = tok_sequence
        self.max_seq_len = max_seq_len

    def __iter__(self):



        for ix in range(self.tok_sequence.shape[0] - self.max_seq_len + 1):

            chunk = torch.from_numpy((self.tok_sequence[ix: ix + self.max_seq_len]).astype(np.int64))
            # if chunk[-1] not in [3, 4]:
            #     continue
            yield chunk

class PretokDatasetJSON(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source, use_ctxmdl_prefix, data_dir, ctxmdl_id):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source
        self.use_ctxmdl_prefix = use_ctxmdl_prefix
        self.bin_dir = data_dir
        self.ctxmdl_id = ctxmdl_id

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        # the .bin files are in tok{N} directory
        if self.bin_dir == None:
            bin_dir = os.path.join(DATA_CACHE_DIR, f"ECM/{self.split}/tok{self.vocab_size}")
        else:
            bin_dir = os.path.join(self.bin_dir, f"{self.split}/tok{self.vocab_size}")

        shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))

        # train/test split. let's use only shard 0 for test split, rest train
        # shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]

        # trainsplit = int(np.floor(len(shard_filenames) * 0.9))
        # shard_filenames = shard_filenames[:trainsplit] if self.split == "train" else shard_filenames[trainsplit:]
        shard_filenames = [s for s in shard_filenames if f"CTX{self.ctxmdl_id}.bin" in s]

        assert len(shard_filenames)>0, f"No bin files found in {bin_dir}"

        num_strides = 32

        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                # generates a binary code for ctx mdl index of length 10:
                if self.use_ctxmdl_prefix:
                    bin_ctx_id = format(int(''.join(filter(str.isdigit, shard.split("/")[-1]))), '010b')
                    bin_ctx_id = bin_ctx_id.replace('1', '3')
                    bin_ctx_id = bin_ctx_id.replace('0', '4')
                    bin_ctx_id_tensor = torch.tensor([int(d) for d in bin_ctx_id])

                m = np.memmap(shard, dtype=np.uint16, mode="r")

                # print(m[:100])

                num_batches = len(m) // self.max_seq_len
                num_batches -= 2  # drop the last partial batch minus the last full batch for sliding operation
                if num_batches < 1:
                    continue
                # assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)

                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = (start + self.max_seq_len + 1) - 10 if self.use_ctxmdl_prefix else (start + self.max_seq_len + 1)
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    for s in range(num_strides):
                        chunk = torch.from_numpy((m[start+s:end+s]).astype(np.int64))
                        if (chunk == 1).any(): # checks if chunk includes a BOS (i.e., new ctx mdl) and avoids this chunk if so
                            continue
                        if self.use_ctxmdl_prefix:
                            chunk = torch.cat((bin_ctx_id_tensor, chunk))

                        x = chunk[:-1]
                        y = chunk[1:]
                        # y = chunk[-1]
                        yield x, y

class PretokDatasetJSONtest(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, tok_sequence, max_seq_len, ctxmdl_id):
        super().__init__()
        self.tok_sequence = tok_sequence
        self.max_seq_len = max_seq_len
        self.ctxmdl_id = ctxmdl_id

    def __iter__(self):
        for ix in range(self.tok_sequence.shape[0] - self.max_seq_len + 1):
            yield self.tok_sequence[ix: ix + self.max_seq_len]

# -----------------------------------------------------------------------------
# public interface functions


class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

    @staticmethod
    def iter_batches_ECM(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDatasetJSON(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

    @staticmethod
    def iter_batches_ECMv2(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDatasetuint16(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

    @staticmethod
    def iter_batches_ECM_test(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDatasetJSONtest(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=False, num_workers=num_workers, drop_last=False
        )
        for x in dl:
            x = x.to(device, non_blocking=True)
            yield x

    @staticmethod
    def iter_batches_ECMv2_test(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDatasetuint16test(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=False, num_workers=num_workers, drop_last=False
        )
        for x in dl:
            x = x.to(device, non_blocking=True)
            yield x



# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["pretokenize_telko_digit", "pretokenize_telko_mtt", "train_telko_vocab", "get_number_of_bins"])
    parser.add_argument("--vocab_size", type=int, default=5, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    parser.add_argument("--data_path", type=str, default="/Users/arndt/Downloads/v2x_parquet")
    parser.add_argument('--out_dir', type=str, default='output')
    parser.add_argument("--tok_train_max_sentences", type=int, default=50)
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "train_telko_vocab":
        os.makedirs(f"{args.out_dir}/tokenizers", exist_ok=True)
        train_telko_vocab(max_sentences=args.tok_train_max_sentences, data_path=args.data_path, out_dir=args.out_dir)
    elif args.stage == "pretokenize_telko_digit":
        pretokenize_telko(split='train', data_path=args.data_path, out_dir=args.out_dir)
        pretokenize_telko(split='test', data_path=args.data_path, out_dir=args.out_dir)
    elif args.stage == "pretokenize_telko_mtt":
        pretokenize_telko(split='train', data_path=args.data_path, out_dir=args.out_dir, normalization=True)
        pretokenize_telko(split='test', data_path=args.data_path, out_dir=args.out_dir, normalization=True)
    elif args.stage == "get_number_of_bins":
        number_of_bins(ctxmdl_id=args.ctxmdl_id, split='train', data_dir=args.data_path)
        number_of_bins(ctxmdl_id=args.ctxmdl_id, split='test', data_dir=args.data_path)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
