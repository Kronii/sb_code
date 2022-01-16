#!/bin/bash

awk 'NR == 1 || NR % 80 == 0' celeb_train_2.csv > celeb_train_2_smola.csv
awk 'NR == 1 || NR % 80 == 0' celeb_val_2.csv > celeb_val_2_smola.csv