NYU NLP Homework 3: Implement a Viterbi HMM POS tagger
    Improved the unknown-word tagging by classify the unknown by common suffixes
    by Ziyang Zeng (zz2960)
    Spring 2022

Pre-requisites:
    - Python 3.8+

How did I handle OOV:
    I did the last suggested approach, which is combination of treating single appearance words as unknown words and classify them into common suffixes.

How to run:
    `python3 main_zz2960_HW3.py --help` will give you:
        usage: main_zz2960_HW3.py [-h] [-s STATEFILE] [-o OUTPUTFILE] mode inputfile

        A Viterbi HMM POS tagger implementation.

        positional arguments:
        mode           "train" or "tag"
        inputfile      input corpus file

        optional arguments:
        -h, --help     show this help message and exit
        -s STATEFILE   path for storing/reading trained state file, default is states.pkl
        -o OUTPUTFILE  path for storing tagged output file, default is output.txt

Examples:
    To train:
        `python3 main_zz2960_HW3.py train data/WSJ_02-21.pos`
    To tag:
        `python3 main_zz2960_HW3.py tag data/WSJ_24.words -o output.txt`
