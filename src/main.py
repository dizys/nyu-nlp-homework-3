import json
import argparse


def train(inputfile, statefile):
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


def tag(inputfile, statefile, outputfile):
    pass


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
