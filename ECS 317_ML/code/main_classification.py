"""
Created on Sunday October 12, 2025, 23:50:38

@author: Tanmay Basu
"""

# from classification2 import classification
from classification3 import classification

import warnings
warnings.filterwarnings("ignore")


clf=classification('/home/Downloads/23060_Arnav_Kapoor/assign1/code/', clf_opt='dt',
                        no_of_selected_features=4)

clf.classification()
