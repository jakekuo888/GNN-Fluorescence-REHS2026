#!/bin/bash

mkdir ./data/absorption-data

sed -i 's/^re_generate_data = False/re_generate_data = True/' ./code/train_test.py

echo "SETUP COMPLETE"