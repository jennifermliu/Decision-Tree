from __future__ import division
from node import Node
import numpy as np
import math
import copy


def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node)
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''

  examplesCopy = copy.deepcopy(examples)

  # if examples is empty
  if len(examplesCopy) == 0:
      tree = Node(None,{})
    #   print "case 1"
    #   print tree.label
      return tree
  # if non-trivial splits
  if checkSplit(examplesCopy) == False:
      tree = Node(mode(examplesCopy,'Class'),{})
    #   print "case 2"
    #   print tree.label
      return tree
  # if all examples have same classification
  sameclass, label = checkSameClass(examplesCopy)
  if sameclass:
      tree = Node(label,{})
    #   print "case 3"
    #   print tree.label
      #print tree.children
      return tree


  # best <- CHOOSE-ATTRIBUTE(examples)
  best = chooseAttribute(examplesCopy)
#   print "case 4"
#   print best

  # t <- new tree with root test best
  t = Node(best,{})
  # for each valuei of best:
  exampleDict, featurePosibilityDict = groupExamples(examplesCopy,best)
#   print exampleDict
  for key in exampleDict:
    #   print key
      # examplesi <- {elements of examples with best = valuei}
      newexamples = exampleDict[key]
    #   print key
      #print newexamples
      # subtree <- ID3(examplesi, MODE(examples)}
      subtree = ID3(newexamples,mode(examplesCopy,'Class'))
      # add branch to t with label valuei and subtree subtree
      (t.children)[key]=subtree
      (t.childrenPosibility)[key] = featurePosibilityDict[key]


  return t



### using postPruningTree
def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  (prune from botton to up)
  '''
  if node.children == {}:
      return node

  examplesCopy = copy.deepcopy(examples)
  classList=[example['Class'] for example in examplesCopy] 
  classWithMajority = majorityClass(classList) # get the label of highest posibility
  labelname = node.label
  exampleDict,_ = groupExamples(examplesCopy,labelname)


  for key in node.children:
      if key in exampleDict:
        copysubTree = copy.deepcopy(node.children[key])
        node.children[key] = prune(copysubTree,exampleDict[key])

  # if splitting is better than non-splitting
  if test(node,examples) >= testingVoting(classWithMajority,examples):
      return node

  # if splitting is worse than non-splitting
  node = Node(classWithMajority,{})
  return node
      

def majorityClass(classList):
    '''
    Takes in a list of labels of data set, return the label with largest quantity/highest posibility
    '''
    classCnt = {}
    for c in classList:
        if c not in classCnt.keys():
            classCnt[c] = 0
        classCnt[c] += 1
    
    return max(classCnt)



def testingVoting(label, examples):
  '''
  Takes label and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  correct = 0
  for example in examples:
      if label == example['Class']:
          correct += 1
  return float(correct)/len(examples)



def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  examplesCopy = copy.deepcopy(examples)
  correct = 0
  for example in examplesCopy:
      result = evaluate(node,example)
      if result == example['Class']:
          correct += 1
  return float(correct)/len(examples)

def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''
  # print example
  # print node.label
  exampleCopy = example
  if node.children == {}:
      return node.label
  if node.label in exampleCopy:
      value = exampleCopy[node.label]
      if value not in node.children:
          value = max(node.childrenPosibility)
      exampleCopy.pop(node.label,None)
  else:
      value = max(node.childrenPosibility)
  return evaluate(node.children[value],exampleCopy)



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

    for attributeName in pairDict:  ### feature
        pairList = pairDict[attributeName]
        attrTolabel = {} # map attribute value to list of labels
        entropySum = 0 # total entropy of all kinds of attribute values
        for pair in pairList:    
            if pair[0] not in attrTolabel:
                attrTolabel[pair[0]]=[]
            attrTolabel[pair[0]].append(pair[1])
        for key in attrTolabel:  ### attribute value of each feature
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
    featurePosibilityDict = {}
    for example in examples:
        if attribute in example:
            value = example.pop(attribute, None)
            # print value
            if value != None:
                if value not in exampleDict:
                    exampleDict[value]=[]
                    featurePosibilityDict[value] = 0
                exampleDict[value].append(example)
                featurePosibilityDict[value] += 1
    return exampleDict,featurePosibilityDict
