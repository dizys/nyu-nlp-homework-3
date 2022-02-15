NYU NLP Homework 3: Implement a Viterbi HMM POS tagger
    by Ziyang Zeng (zz2960)
    Spring 2022

Pre-requisites:
    - Python 3.7+

How to run:
    `python3 main_zz2960_HW3.py --help` will give you:
        usage: main_zz2960_HW3.py [-h] [-s STATEFILE] [-o OUTPUTFILE] mode inputfile

        A Viterbi HMM POS tagger implementation.

        positional arguments:
        mode           "train" or "tag"
        inputfile      input corpus file

        optional arguments:
        -h, --help     show this help message and exit
        -s STATEFILE   path for storing/reading trained state file, default is states.json
        -o OUTPUTFILE  path for storing tagged output file, default is output.txt

Examples:
    To train:
        `python3 main_zz2960_HW3.py train data/WSJ_02-21.pos`
    To tag:
        `python3 main_zz2960_HW3.py tag data/WSJ_24.words -o output.txt`
