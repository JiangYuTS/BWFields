import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def Read_GLIMPSE_Bubbles(file_name):
    bubble_names = []
    bubble_ls = []
    bubble_bs = []
    bubble_Rrs = [[],[],[],[],[]]
    df = pd.read_excel(file_name,header=1)
    bubble_lines = df.values
    for i in range(0,133):
        bubble_line = bubble_lines[i][0].split()
        bubble_name = ''
        for item_id in bubble_line[0]:
            if item_id != '.':
                bubble_name += item_id
            else:
                break
        bubble_names.append(bubble_name)
        bubble_ls.append(float(bubble_line[1]))
        bubble_b = ''
        for item_b in bubble_line[2]:
            if item_b=='?':
                bubble_b += '-'
            else:
                bubble_b += item_b
        bubble_bs.append(float(bubble_b))
    #     for item_Rin,item_rin,item_Rout,item_rout in zip(bubble_line[3],bubble_line[4],bubble_line[5],bubble_line[6]):
        bubble_Rrs[0].append(float(bubble_line[3]))
        bubble_Rrs[1].append(float(bubble_line[4]))
        bubble_Rrs[2].append(float(bubble_line[5]))
        bubble_Rrs[3].append(float(bubble_line[6]))
        bubble_Rrs[4].append(float(bubble_line[8]))
    bubble_centers_wcs = np.c_[bubble_ls,bubble_bs]
    return bubble_names,bubble_ls,bubble_bs,bubble_centers_wcs,bubble_Rrs




