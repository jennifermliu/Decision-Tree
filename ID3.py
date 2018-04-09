from __future__ import division
from node import Node
import numpy as np
import math


def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node)
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''

  # print examples

  # if examples is empty
  if len(examples) == 0:
      tree = Node(None,{})
      #print tree.label
      return tree
  # if non-trivial splits
  if checkSplit(examples) == False:
      tree = Node(mode(examples,'Class'),{})
      #print tree.label
      return tree
  # if all examples have same classification
  sameclass, label = checkSameClass(examples)
  if sameclass:
      tree = Node(label,{})
      #print tree.label
      #print tree.children
      return tree


  # best <- CHOOSE-ATTRIBUTE(examples)
  best = chooseAttribute(examples)

  # t <- new tree with root test best
  t = Node(best,{})
  # for each valuei of best:
  exampleDict = groupExamples(examples,best)
  for key in exampleDict:
      # examplesi <- {elements of examples with best = valuei}
      newexamples = exampleDict[key]
    #   print key
      #print newexamples
      # subtree <- ID3(examplesi, MODE(examples)}
      subtree = ID3(newexamples,mode(examples,'Class'))
      # add branch to t with label valuei and subtree subtree
      (t.children)[key]=subtree


  return t



def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''

def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''
  # print example
  # print node.label
  if node.children == {}:
      return node.label
  if node.label in example:
      value = example[node.label]
      example.pop(node.label,None)
     # print example
      return evaluate(node.children[value],example)




# return true if all examples belong to the same class and return the class value
def checkSameClass(examples):
    curr = None
    for example in examples:
        label = example.get('Class', None)
        if label != None:
            if curr == None:
                curr = label
            elif label != curr:
                return False, curr
    return True, curr

# return true if there's another split on attributes
def checkSplit(examples):
    example = examples[0]
    if len(example) < 2:
        return False
    return True

# return the most frequent value for this attribute
def mode(examples,attribute):
    count = {}
    for example in examples:
        # print example
        label = example.get(attribute, None)
        # print label
        if label != None:
            if label in count:
                count[label] = count[label] + 1
            else:
                count[label] = 1
    highest = 0
    best = None
    for key, value in count.iteritems():
        if value > highest:
            highest = value
            best = key
    return best

# return the best attribute to split on
def chooseAttribute(examples):
    best = None # best attribute to split on
    lowest = float("inf")
    pairDict = {} # map attribute name to list of attributeValue/label pairs
    for example in examples:
        for key in example:
            if key != 'Class':
                if key not in pairDict:
                    pairDict[key]=[]
                pairDict[key].append([example[key],example['Class']])

    for attributeName in pairDict:
        pairList = pairDict[attributeName]
        attrTolabel = {} # map attribute value to list of labels
        entropySum = 0 # total entropy of all kinds of attribute values
        for pair in pairList:
            if pair[0] not in attrTolabel:
                attrTolabel[pair[0]]=[]
            attrTolabel[pair[0]].append(pair[1])
        for key in attrTolabel:
            # map label value to its count within this attribute value
            labelCount = {}
            totalCount = 0 # total count of examples with this attribute value
            labelList = attrTolabel[key]
            for label in labelList:
                if label not in labelCount:
                    labelCount[label] = 0
                labelCount[label] += 1
                totalCount += 1
            groupEntropy = 0 # total entropy of all kinds of labels with this attribute value
            for label in labelCount:
                count = labelCount[label]
                entropy = -count/totalCount * np.log2(count/totalCount)
                groupEntropy += entropy
            entropySum = totalCount/len(pairDict) * groupEntropy
        if entropySum < lowest:
            lowest = entropySum
            best = attributeName
    return best

# return dict mapping from attribute value to new examples without this attribute
def groupExamples(examples,attribute):
    exampleDict = {}
    for example in examples:
        if attribute in example:
            value = example.pop(attribute, None)
            # print value
            if value != None:
                if value not in exampleDict:
                    exampleDict[value]=[]
                exampleDict[value].append(example)
    return exampleDict
