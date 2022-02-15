#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NYU NLP Homework 3: Implement a Viterbi HMM POS tagger
    by Ziyang Zeng (zz2960)
    Spring 2022
"""

import json
import argparse

from typing import Dict, List


class ViterbiHMMPOSTagger:
    def __init__(self, word_tag_count: Dict[str, Dict[str, int]], tag_count: Dict[str, int], tag_tag_count: Dict[str, Dict[str, int]]):
        self.word_tag_count = word_tag_count
        self.tag_count = tag_count
        self.tag_tag_count = tag_tag_count

    def tag(self, sentence) -> List[str]:
        word_tag_count = self.word_tag_count
        tag_count = self.tag_count
        tag_tag_count = self.tag_tag_count

        # Initialize the viterbi table
        viterbi_table = {}
        for word in sentence:
            for tag in tag_count:
                if word not in word_tag_count:
                    viterbi_table[(word, tag)] = 0
                else:
                    viterbi_table[(word, tag)] = 0

        # Initialize the backpointer table
        backpointer_table = {}
        for word in sentence:
            for tag in tag_count:
                backpointer_table[(word, tag)] = ''

        # Initialize the first row
        for tag in tag_count:
            if sentence[0] in word_tag_count:
                viterbi_table[(sentence[0], tag)
                              ] = word_tag_count[sentence[0]][tag] / tag_count[tag]
            else:
                viterbi_table[(sentence[0], tag)] = 0

        # Iterate over the rest of the rows
        for i in range(1, len(sentence)):
            for tag in tag_count:
                max_prob = 0
                max_tag = ''
                for prev_tag in tag_count:
                    if (sentence[i], tag) in word_tag_count:
                        prob = viterbi_table[(sentence[i], prev_tag)] * tag_tag_count[prev_tag][tag] * \
                            word_tag_count[sentence[i]][tag] / \
                            tag_count[prev_tag]
                    else:
                        prob = viterbi_table[(
                            sentence[i], prev_tag)] * tag_tag_count[prev_tag][tag] / tag_count[prev_tag]
                    if prob > max_prob:
                        max_prob = prob
                        max_tag = prev_tag
                viterbi_table[(sentence[i], tag)] = max_prob
                backpointer_table[(sentence[i], tag)] = max_tag

        # Find the best path
        max_prob = 0
        max_tag = ''
        for tag in tag_count:
            if viterbi_table[(sentence[-1], tag)] > max_prob:
                max_prob = viterbi_table[(sentence[-1], tag)]
                max_tag = tag
        tag_sequence = [max_tag]
        for i in range(len(sentence) - 1, 0, -1):
            tag_sequence.append(
                backpointer_table[(sentence[i], tag_sequence[-1])])
        tag_sequence.reverse()
        return tag_sequence


def train(inputfile: str, statefile: str) -> None:
    word_tag_count = {}
    tag_count = {}
    tag_tag_count = {}
    with open(inputfile, "r", encoding="utf-8") as f:
        last_tag = 'B'
        tag_count['B'] = tag_count.get('B', 0) + 1
        for line in f:
            line = line.strip()
            if line:
                word, tag = [x.strip() for x in line.split("\t")]
                if word not in word_tag_count:
                    word_tag_count[word] = {}
                if tag not in word_tag_count[word]:
                    word_tag_count[word][tag] = 0
                word_tag_count[word][tag] += 1
                tag_count[tag] = tag_count.get(tag, 0) + 1
                if last_tag not in tag_tag_count:
                    tag_tag_count[last_tag] = {}
                if tag not in tag_tag_count[last_tag]:
                    tag_tag_count[last_tag][tag] = 0
                tag_tag_count[last_tag][tag] += 1
                last_tag = tag
            else:
                tag = 'E'
                if last_tag not in tag_tag_count:
                    tag_tag_count[last_tag] = {}
                if tag not in tag_tag_count[last_tag]:
                    tag_tag_count[last_tag][tag] = 0
                tag_count['E'] = tag_count.get('E', 0) + 1
                tag_tag_count[last_tag][tag] += 1
                tag_count['B'] = tag_count.get('B', 0) + 1
                last_tag = 'B'
    print(f"{len(word_tag_count)} words and {len(tag_count)} tags.")
    # store states to file
    states = {
        'word_tag_count': word_tag_count,
        'tag_count': tag_count,
        'tag_tag_count': tag_tag_count
    }
    with open(statefile, "w") as f:
        json.dump(states, f)


def tag(inputfile: str, statefile: str, outputfile: str) -> None:
    with open(statefile, "r") as f:
        states = json.load(f)
    tagger = ViterbiHMMPOSTagger(
        states['word_tag_count'], states['tag_count'], states['tag_tag_count'])
    with open(inputfile, "r", encoding="utf-8") as f:
        with open(outputfile, "w", encoding="utf-8") as out:
            sentence = []
            for line in f:
                line = line.strip()
                if line:
                    items = [x.strip() for x in line.split("\t")]
                    sentence.append(items[0])
                else:
                    tag_sequence = tagger.tag(sentence)
                    for word, tag in zip(sentence, tag_sequence):
                        out.write(f"{word}\t{tag}\n")
                    out.write("\n")
                    sentence = []
            if len(sentence) > 0:
                tag_sequence = tagger.tag(sentence)
                for word, tag in zip(sentence, tag_sequence):
                    out.write(f"{word}\t{tag}\n")
    print(f"{inputfile} -> {outputfile}")


def main():
    parser = argparse.ArgumentParser(
        description="A Viterbi HMM POS tagger implementation.")
    parser.add_argument("mode", help="\"train\" or \"tag\"")
    parser.add_argument("inputfile", help="input corpus file")
    parser.add_argument(
        "-s", dest="statefile", help="path for storing/reading trained state file, default is states.json", default="states.json")
    parser.add_argument(
        "-o", dest="outputfile", help="path for storing tagged output file, default is output.txt", default="output.txt")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.inputfile, args.statefile)
    elif args.mode == "tag":
        tag(args.inputfile, args.statefile, args.outputfile)
    else:
        print(f"Invalid mode: {args.mode}. Should be \"train\" or \"tag\".")
        exit(1)


if __name__ == '__main__':
    main()
