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
        viterbi_table = [{'B': 1}]
        backpointer_table = [{'B': 'B'}]

        for word in sentence:
            last_viterbi_row = viterbi_table[-1]
            new_viterbi_row = {}
            new_backpointer_row = {}

            for tag in tag_count:
                max_last_tag = None
                max_prob = -1
                for last_tag in last_viterbi_row:
                    last_tag_prob = last_viterbi_row[last_tag]
                    if last_tag in tag_tag_count and tag in tag_tag_count[last_tag]:
                        trans_prob = tag_tag_count[last_tag][tag] / \
                            tag_count[last_tag]
                    else:
                        trans_prob = 0
                    if word in word_tag_count and tag in word_tag_count[word]:
                        emission_prob = word_tag_count[word][tag] / \
                            tag_count[tag]
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
            if last_tag in tag_tag_count and 'E' in tag_tag_count[last_tag]:
                trans_prob = tag_tag_count[last_tag]['E'] / tag_count[last_tag]
            else:
                trans_prob = 0
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


def train(inputfile: str, statefile: str) -> None:
    word_tag_count = {}
    tag_count = {}
    tag_tag_count = {}
    with open(inputfile, "r", encoding="utf-8") as f:
        last_tag = 'B'
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
