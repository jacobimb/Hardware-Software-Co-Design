#!/bin/bash

# FFCN
conda run -n thesis_env python evaluation.py -a 0 -c 0 >> ./logs/logfile_0_0_eval.txt
conda run -n thesis_env python evaluation.py -a 0 -c 1 >> ./logs/logfile_0_1_eval.txt
conda run -n thesis_env python evaluation.py -a 0 -c 2 >> ./logs/logfile_0_2_eval.txt
conda run -n thesis_env python evaluation.py -a 0 -c 3 >> ./logs/logfile_0_3_eval.txt
conda run -n thesis_env python evaluation.py -a 0 -c 4 >> ./logs/logfile_0_4_eval.txt

# DSCNN
conda run -n thesis_env python evaluation.py -a 1 -c 0 >> ./logs/logfile_1_0_eval.txt
conda run -n thesis_env python evaluation.py -a 1 -c 1 >> ./logs/logfile_1_1_eval.txt
conda run -n thesis_env python evaluation.py -a 1 -c 2 >> ./logs/logfile_1_2_eval.txt
conda run -n thesis_env python evaluation.py -a 1 -c 3 >> ./logs/logfile_1_3_eval.txt
conda run -n thesis_env python evaluation.py -a 1 -c 4 >> ./logs/logfile_1_4_eval.txt
conda run -n thesis_env python evaluation.py -a 1 -c 5 >> ./logs/logfile_1_5_eval.txt
conda run -n thesis_env python evaluation.py -a 1 -c 6 >> ./logs/logfile_1_6_eval.txt
conda run -n thesis_env python evaluation.py -a 1 -c 7 >> ./logs/logfile_1_7_eval.txt
conda run -n thesis_env python evaluation.py -a 1 -c 8 >> ./logs/logfile_1_8_eval.txt

# DenseNet
conda run -n thesis_env python evaluation.py -a 2 -c 0 >> ./logs/logfile_2_0_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 1 >> ./logs/logfile_2_1_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 2 >> ./logs/logfile_2_2_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 3 >> ./logs/logfile_2_3_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 4 >> ./logs/logfile_2_4_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 5 >> ./logs/logfile_2_5_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 6 >> ./logs/logfile_2_6_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 7 >> ./logs/logfile_2_7_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 8 >> ./logs/logfile_2_8_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 9 >> ./logs/logfile_2_9_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 10 >> ./logs/logfile_2_10_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 11 >> ./logs/logfile_2_11_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 12 >> ./logs/logfile_2_12_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 13 >> ./logs/logfile_2_13_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 14 >> ./logs/logfile_2_14_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 15 >> ./logs/logfile_2_15_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 16 >> ./logs/logfile_2_16_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 17 >> ./logs/logfile_2_17_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 18 >> ./logs/logfile_2_18_eval.txt
conda run -n thesis_env python evaluation.py -a 2 -c 19 >> ./logs/logfile_2_19_eval.txt

# ResNet
conda run -n thesis_env python evaluation.py -a 3 -c 0 >> ./logs/logfile_3_0_eval.txt
conda run -n thesis_env python evaluation.py -a 3 -c 1 >> ./logs/logfile_3_1_eval.txt
conda run -n thesis_env python evaluation.py -a 3 -c 2 >> ./logs/logfile_3_2_eval.txt
conda run -n thesis_env python evaluation.py -a 3 -c 3 >> ./logs/logfile_3_3_eval.txt
conda run -n thesis_env python evaluation.py -a 3 -c 4 >> ./logs/logfile_3_4_eval.txt
conda run -n thesis_env python evaluation.py -a 3 -c 5 >> ./logs/logfile_3_5_eval.txt

# Inception
conda run -n thesis_env python evaluation.py -a 4 -c 0 >> ./logs/logfile_4_0_eval.txt
conda run -n thesis_env python evaluation.py -a 4 -c 1 >> ./logs/logfile_4_1_eval.txt
conda run -n thesis_env python evaluation.py -a 4 -c 2 >> ./logs/logfile_4_2_eval.txt
conda run -n thesis_env python evaluation.py -a 4 -c 3 >> ./logs/logfile_4_3_eval.txt
conda run -n thesis_env python evaluation.py -a 4 -c 4 >> ./logs/logfile_4_4_eval.txt
conda run -n thesis_env python evaluation.py -a 4 -c 5 >> ./logs/logfile_4_5_eval.txt
conda run -n thesis_env python evaluation.py -a 4 -c 6 >> ./logs/logfile_4_6_eval.txt
conda run -n thesis_env python evaluation.py -a 4 -c 7 >> ./logs/logfile_4_7_eval.txt

# CENet
conda run -n thesis_env python evaluation.py -a 5 -c 0 >> ./logs/logfile_5_0_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 1 >> ./logs/logfile_5_1_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 2 >> ./logs/logfile_5_2_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 3 >> ./logs/logfile_5_3_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 4 >> ./logs/logfile_5_4_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 5 >> ./logs/logfile_5_5_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 6 >> ./logs/logfile_5_6_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 7 >> ./logs/logfile_5_7_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 8 >> ./logs/logfile_5_8_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 9 >> ./logs/logfile_5_9_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 10 >> ./logs/logfile_5_10_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 11 >> ./logs/logfile_5_11_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 12 >> ./logs/logfile_5_12_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 13 >> ./logs/logfile_5_13_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 14 >> ./logs/logfile_5_14_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 15 >> ./logs/logfile_5_15_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 16 >> ./logs/logfile_5_16_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 17 >> ./logs/logfile_5_17_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 18 >> ./logs/logfile_5_18_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 19 >> ./logs/logfile_5_19_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 20 >> ./logs/logfile_5_20_eval.txt
conda run -n thesis_env python evaluation.py -a 5 -c 21 >> ./logs/logfile_5_21_eval.txt
