#!/bin/bash

# FFCN
conda run -n thesis_env python main.py -a 0 -c 0 >> ./logs/logfile_0_0.txt
conda run -n thesis_env python main.py -a 0 -c 1 >> ./logs/logfile_0_1.txt
conda run -n thesis_env python main.py -a 0 -c 2 >> ./logs/logfile_0_2.txt
conda run -n thesis_env python main.py -a 0 -c 3 >> ./logs/logfile_0_3.txt
conda run -n thesis_env python main.py -a 0 -c 4 >> ./logs/logfile_0_4.txt

# DSCNN
conda run -n thesis_env python main.py -a 1 -c 0 >> ./logs/logfile_1_0.txt
conda run -n thesis_env python main.py -a 1 -c 1 >> ./logs/logfile_1_1.txt
conda run -n thesis_env python main.py -a 1 -c 2 >> ./logs/logfile_1_2.txt
conda run -n thesis_env python main.py -a 1 -c 3 >> ./logs/logfile_1_3.txt
conda run -n thesis_env python main.py -a 1 -c 4 >> ./logs/logfile_1_4.txt
conda run -n thesis_env python main.py -a 1 -c 5 >> ./logs/logfile_1_5.txt
conda run -n thesis_env python main.py -a 1 -c 6 >> ./logs/logfile_1_6.txt
conda run -n thesis_env python main.py -a 1 -c 7 >> ./logs/logfile_1_7.txt
conda run -n thesis_env python main.py -a 1 -c 8 >> ./logs/logfile_1_8.txt

# DenseNet
conda run -n thesis_env python main.py -a 2 -c 0 >> ./logs/logfile_2_0.txt
conda run -n thesis_env python main.py -a 2 -c 1 >> ./logs/logfile_2_1.txt
conda run -n thesis_env python main.py -a 2 -c 2 >> ./logs/logfile_2_2.txt
conda run -n thesis_env python main.py -a 2 -c 3 >> ./logs/logfile_2_3.txt
conda run -n thesis_env python main.py -a 2 -c 4 >> ./logs/logfile_2_4.txt
conda run -n thesis_env python main.py -a 2 -c 5 >> ./logs/logfile_2_5.txt
conda run -n thesis_env python main.py -a 2 -c 6 >> ./logs/logfile_2_6.txt
conda run -n thesis_env python main.py -a 2 -c 7 >> ./logs/logfile_2_7.txt
conda run -n thesis_env python main.py -a 2 -c 8 >> ./logs/logfile_2_8.txt
conda run -n thesis_env python main.py -a 2 -c 9 >> ./logs/logfile_2_9.txt
conda run -n thesis_env python main.py -a 2 -c 10 >> ./logs/logfile_2_10.txt
conda run -n thesis_env python main.py -a 2 -c 11 >> ./logs/logfile_2_11.txt
conda run -n thesis_env python main.py -a 2 -c 12 >> ./logs/logfile_2_12.txt
conda run -n thesis_env python main.py -a 2 -c 13 >> ./logs/logfile_2_13.txt
conda run -n thesis_env python main.py -a 2 -c 14 >> ./logs/logfile_2_14.txt
conda run -n thesis_env python main.py -a 2 -c 15 >> ./logs/logfile_2_15.txt
conda run -n thesis_env python main.py -a 2 -c 16 >> ./logs/logfile_2_16.txt
conda run -n thesis_env python main.py -a 2 -c 17 >> ./logs/logfile_2_17.txt
conda run -n thesis_env python main.py -a 2 -c 18 >> ./logs/logfile_2_18.txt
conda run -n thesis_env python main.py -a 2 -c 19 >> ./logs/logfile_2_19.txt

# ResNet
conda run -n thesis_env python main.py -a 3 -c 0 >> ./logs/logfile_3_0.txt
conda run -n thesis_env python main.py -a 3 -c 1 >> ./logs/logfile_3_1.txt
conda run -n thesis_env python main.py -a 3 -c 2 >> ./logs/logfile_3_2.txt
conda run -n thesis_env python main.py -a 3 -c 3 >> ./logs/logfile_3_3.txt
conda run -n thesis_env python main.py -a 3 -c 4 >> ./logs/logfile_3_4.txt
conda run -n thesis_env python main.py -a 3 -c 5 >> ./logs/logfile_3_5.txt

# Inception
conda run -n thesis_env python main.py -a 4 -c 0 >> ./logs/logfile_4_0.txt
conda run -n thesis_env python main.py -a 4 -c 1 >> ./logs/logfile_4_1.txt
conda run -n thesis_env python main.py -a 4 -c 2 >> ./logs/logfile_4_2.txt
conda run -n thesis_env python main.py -a 4 -c 3 >> ./logs/logfile_4_3.txt
conda run -n thesis_env python main.py -a 4 -c 4 >> ./logs/logfile_4_4.txt
conda run -n thesis_env python main.py -a 4 -c 5 >> ./logs/logfile_4_5.txt
conda run -n thesis_env python main.py -a 4 -c 6 >> ./logs/logfile_4_6.txt
conda run -n thesis_env python main.py -a 4 -c 7 >> ./logs/logfile_4_7.txt

# CENet
conda run -n thesis_env python main.py -a 5 -c 0 >> ./logs/logfile_5_0.txt
conda run -n thesis_env python main.py -a 5 -c 1 >> ./logs/logfile_5_1.txt
conda run -n thesis_env python main.py -a 5 -c 2 >> ./logs/logfile_5_2.txt
conda run -n thesis_env python main.py -a 5 -c 3 >> ./logs/logfile_5_3.txt
conda run -n thesis_env python main.py -a 5 -c 4 >> ./logs/logfile_5_4.txt
conda run -n thesis_env python main.py -a 5 -c 5 >> ./logs/logfile_5_5.txt
conda run -n thesis_env python main.py -a 5 -c 6 >> ./logs/logfile_5_6.txt
conda run -n thesis_env python main.py -a 5 -c 7 >> ./logs/logfile_5_7.txt
conda run -n thesis_env python main.py -a 5 -c 8 >> ./logs/logfile_5_8.txt
conda run -n thesis_env python main.py -a 5 -c 9 >> ./logs/logfile_5_9.txt
conda run -n thesis_env python main.py -a 5 -c 10 >> ./logs/logfile_5_10.txt
conda run -n thesis_env python main.py -a 5 -c 11 >> ./logs/logfile_5_11.txt
conda run -n thesis_env python main.py -a 5 -c 12 >> ./logs/logfile_5_12.txt
conda run -n thesis_env python main.py -a 5 -c 13 >> ./logs/logfile_5_13.txt
conda run -n thesis_env python main.py -a 5 -c 14 >> ./logs/logfile_5_14.txt
conda run -n thesis_env python main.py -a 5 -c 15 >> ./logs/logfile_5_15.txt
conda run -n thesis_env python main.py -a 5 -c 16 >> ./logs/logfile_5_16.txt
conda run -n thesis_env python main.py -a 5 -c 17 >> ./logs/logfile_5_17.txt
conda run -n thesis_env python main.py -a 5 -c 18 >> ./logs/logfile_5_18.txt
conda run -n thesis_env python main.py -a 5 -c 19 >> ./logs/logfile_5_19.txt
conda run -n thesis_env python main.py -a 5 -c 20 >> ./logs/logfile_5_20.txt
conda run -n thesis_env python main.py -a 5 -c 21 >> ./logs/logfile_5_21.txt