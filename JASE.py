from __future__ import division
from collections import Counter
import re




def load_counts(filename, sep='\t'):
    """
    :param filename: name of file .txt
    :param sep: the separator between words and numbers in the .txt corpus
    :return: a Counter initialized from key-value pairs,
    one on each line of filename
    """

    C = Counter()
    for line in open(filename):
        if line =="EOF":
            break
        key, count = line.split(sep)
        count = re.sub("\D", "", count)
        C[key] = int(count)


    return C


def pdist(counter):
    """
    :param counter: the dict counter of the words
    :return: a probability distribution over the corpus
    """
    N = sum(counter.values())
    return lambda x: counter[x]/N


COUNTS1 = load_counts('count_1w.txt')
COUNTS2 = load_counts('count_2w.txt')

P1w = pdist(COUNTS1)
P2w = pdist(COUNTS2)



def Pwords2(words, prev='<S>'):
    """
    :param words: the words
    :param prev: the previous word
    :return: The probability of a sequence of words, using bigram data, given prev word
    """
    return product(cPword(w, (prev if (i == 0) else words[i-1]) )
                   for (i, w) in enumerate(words))


def cPword(word, prev):
    """
    :param word: the current word
    :param prev: the previous word
    :return: Conditional probability of word, given previous word
    """
    bigram = prev + ' ' + word
    if P2w(bigram) > 0 and P1w(prev) > 0:
        return P2w(bigram) / P1w(prev)
    else:  # Average the back-off value and zero.
        return P1w(word) / 2

def product(nums):
    """
    :param nums: numbers to multiply together
    :return: the product
    """
    result = 1
    for x in nums:
        result *= x
    return result



def splits(text, start=0, L=20):
    """
    :param text: the text to split
    :param start: where to start
    :param L: max length of splitting
    :return: Return a list of all (first, rest) pairs; start <= len(first) <= L
    """

    return [(text[:i], text[i:])
            for i in range(start, min(len(text), L)+1)]

def segment2(text, prev='<S>'):
    """
    :param text: text to split
    :param prev: the previous word splitted
    :return: the best segmentation of text; use bigram data
    """

    if not text:
        return []
    else:
        candidates = ([first] + segment2(rest, first)
                      for (first, rest) in splits(text, 1))
        return max(candidates, key=lambda words: Pwords2(words, prev))


def most_prob_sent(sentences):
    """
    :param sentences: all possible sequences of NOT separated words
    :return: the most probable sentence separated
    """
    for i, el in enumerate(sentences):
        sentences[i] = segment2(el)
    prob = 0
    most_prob_sentence = ""
    for sentence in sentences:
        value = 1
        for word in sentence:
            value = value * P1w(word)
        if value > prob:
            prob = value
            most_prob_sentence = sentence

    return most_prob_sentence

