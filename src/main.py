#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NYU NLP Homework 3: Implement a Viterbi HMM POS tagger
    by Ziyang Zeng (zz2960)
    Spring 2022
"""

import argparse
import pickle

from typing import Dict, List, Tuple, Set, Union, TypedDict

suffixes: List[Union[List[str], str]] = [['able', 'ible'], 'al', 'an', 'ar', 'ed', 'en', ['er', 'or'],
                                         'est', 'ing', ['ish', 'ous', 'ful', 'less'], 'ive', 'ly', ['ment', 'ness'], 'y']


def get_unknown_word_class_by_suffix(word: str) -> str:
    word_class = "UNKNOWN"
    for suffix in suffixes:
        if type(suffix) == list:
            word_suffix_class = suffix[0].upper()
            selected = False
            for suffix_item in suffix:
                if word.endswith(suffix_item):
                    word_class = f"UNKNOWN_AFFIXED_WITH_{word_suffix_class}"
                    selected = True
                    break
            if selected:
                break
        else:
            if word.endswith(suffix):
                word_class = f"UNKNOWN_AFFIXED_WITH_{suffix.upper()}"
                break
    return word_class


class TrainedStates(TypedDict):
    tags: Set[str]
    trans_prob: Dict[Tuple[str, str], float]  # (tag1, tag2) -> prob
    emission_prob: Dict[Tuple[str, str], float]  # (word, tag) -> prob


class ViterbiHMMPOSTagger:
    def __init__(self, states: TrainedStates):
        self.states = states

    def tag(self, sentence) -> List[str]:
        trans_tb = self.states["trans_prob"]
        emission_tb = self.states["emission_prob"]
        tag_set = self.states["tags"]

        # Initialize the viterbi table
        viterbi_table = [{'B': 1}]
        backpointer_table = [{'B': 'B'}]

        for word in sentence:
            word = word.lower()
            last_viterbi_row = viterbi_table[-1]
            new_viterbi_row = {}
            new_backpointer_row = {}

            for tag in tag_set:
                max_last_tag = None
                max_prob = -1
                for last_tag in last_viterbi_row:
                    last_tag_prob = last_viterbi_row[last_tag]
                    trans_prob = trans_tb[(last_tag, tag)]
                    if (word, tag) in emission_tb:
                        emission_prob = emission_tb[(word, tag)]
                    else:
                        word_class = get_unknown_word_class_by_suffix(word)
                        if (word_class, tag) in emission_tb:
                            emission_prob = emission_tb[(word_class, tag)]
                        else:
                            emission_prob = 1 / 1000
                    prob = last_tag_prob * trans_prob * emission_prob
                    if prob > max_prob:
                        max_last_tag = last_tag
                        max_prob = prob
                if max_last_tag:
                    new_viterbi_row[tag] = max_prob
                    new_backpointer_row[tag] = max_last_tag

            viterbi_table.append(new_viterbi_row)
            backpointer_table.append(new_backpointer_row)

        # End of sentence
        last_viterbi_row = viterbi_table[-1]
        max_last_tag = None
        max_prob = -1
        for last_tag in last_viterbi_row:
            last_tag_prob = last_viterbi_row[last_tag]
            trans_prob = trans_tb[(last_tag, 'E')]
            prob = last_tag_prob * trans_prob
            if prob > max_prob:
                max_last_tag = last_tag
                max_prob = prob

        if not max_last_tag:
            return []
        # find the best path back
        tags = [max_last_tag]
        for i in range(len(backpointer_table) - 2):
            max_last_tag = backpointer_table[-i - 1][max_last_tag]
            tags.insert(0, max_last_tag)
        return tags


class BasicStatistics(TypedDict):
    word_tag_count: Dict[str, Dict[str, int]]
    tag_set: Set[str]
    word_count: Dict[str, int]
    tag_count: Dict[str, int]
    tag_tag_count: Dict[str, Dict[str, int]]


def train_get_statistics(lines: List[str]) -> 'BasicStatistics':
    word_tag_count: Dict[Tuple[str, str], int] = {}
    tag_set: Set[str] = set(['B', 'E'])
    word_count: Dict[str, int] = {}
    tag_count: Dict[str, int] = {}
    tag_tag_count: Dict[Tuple[str, str], int] = {}

    last_tag = 'B'
    for line in lines:
        if last_tag == 'B':
            tag_count['B'] = tag_count.get('B', 0) + 1

        line = line.strip()
        if line:
            word, tag = [x.strip() for x in line.split("\t")]
            word = word.lower()
        else:
            word = ''
            tag = 'E'

        tag_count[tag] = tag_count.get(tag, 0) + 1
        tag_tag_count[(last_tag, tag)] = tag_tag_count.get(
            (last_tag, tag), 0) + 1
        if word:
            word_count[word] = word_count.get(word, 0) + 1
            tag_set.add(tag)
            word_tag_count[(word, tag)] = word_tag_count.get(
                (word, tag), 0) + 1

        if tag != 'E':
            last_tag = tag
        else:
            last_tag = 'B'
    return BasicStatistics(word_tag_count=word_tag_count, tag_set=tag_set, word_count=word_count, tag_count=tag_count, tag_tag_count=tag_tag_count)


def train_unkownify_statistics(statistics: BasicStatistics) -> BasicStatistics:
    word_tag_count = statistics["word_tag_count"]
    word_count = statistics["word_count"].copy()

    for word, count in statistics["word_count"].items():
        if count > 1:
            continue
        word_class = get_unknown_word_class_by_suffix(word)
        word_count.pop(word)
        word_count[word_class] = word_count.get(word_class, 0) + 1
        for tag in statistics["tag_count"].keys():
            original_word_tag_count = word_tag_count.pop((word, tag), 0)
            word_tag_count[(word_class, tag)] = word_tag_count.get(
                (word_class, tag), 0) + original_word_tag_count

    return BasicStatistics(word_tag_count=word_tag_count, tag_set=statistics["tag_set"], word_count=word_count, tag_count=statistics["tag_count"], tag_tag_count=statistics["tag_tag_count"])


def train_get_trans_prob(tag_count: Dict[str, int], tag_tag_count: Dict[Tuple[str, str], int]) -> Dict[Tuple[str, str], float]:
    trans_prob: Dict[Tuple[str, str], float] = {}  # (tag1, tag2) -> prob
    for tag1 in tag_count.keys():
        for tag2 in tag_count.keys():
            total_tag1 = tag_count.get(tag1, 0)
            tag1_followed_by_tag2 = tag_tag_count.get((tag1, tag2), 0)
            if total_tag1 == 0:
                prob = 0
            else:
                prob = tag1_followed_by_tag2 / total_tag1
            trans_prob[(tag1, tag2)] = prob
    return trans_prob


def train_get_emission_prob(word_count: Dict[str, int], tag_count: Dict[str, int], word_tag_count: Dict[Tuple[str, str], int]) -> Dict[Tuple[str, str], float]:
    emission_prob: Dict[Tuple[str, str], float] = {}  # (word, tag) -> prob
    for word in word_count.keys():
        for tag in tag_count.keys():
            word_tag = word_tag_count.get((word, tag), 0)
            tag_total = tag_count.get(tag, 0)
            if tag_total == 0:
                prob = 0
            else:
                prob = word_tag / tag_total
            emission_prob[(word, tag)] = prob
    return emission_prob


def train(inputfile: str, statefile: str) -> None:
    with open(inputfile, 'r') as f:
        lines = f.readlines()
    statistics = train_get_statistics(lines)
    statistics = train_unkownify_statistics(statistics)
    trans_prob = train_get_trans_prob(
        statistics["tag_count"], statistics["tag_tag_count"])
    emission_prob = train_get_emission_prob(
        statistics["word_count"], statistics["tag_count"], statistics["word_tag_count"])
    states = TrainedStates(
        trans_prob=trans_prob, emission_prob=emission_prob, tags=statistics["tag_set"])
    print(
        f"{len(statistics['word_count'])} distinct words, {len(statistics['tag_count'])} tags")
    print(f"{len(trans_prob)} trans_prob pairs, {len(emission_prob)} emission pairs.")
    with open(statefile, "wb") as f:
        pickle.dump(states, f)
        print(f"States -> {statefile}")


def tag(inputfile: str, statefile: str, outputfile: str) -> None:
    with open(statefile, "rb") as f:
        states: TrainedStates = pickle.load(f)
    tagger = ViterbiHMMPOSTagger(states)
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
        "-s", dest="statefile", help="path for storing/reading trained state file, default is states.pkl", default="states.pkl")
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
