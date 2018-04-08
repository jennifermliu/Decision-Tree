from node import Node
import math

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node)
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''

  print examples

  # if examples is empty
  if len(examples) == 0:
      tree = Node(None,{})
      return tree
  # if all examples have same classification
  sameclass, label = checkSameClass(examples)
  if sameclass:
      tree = Node(label,{})
      return tree
  # if non-trivial splits
  if checkSplit(examples) == False:
      tree = Node(mode(examples,'Class'),{})
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
    example = examples[0]
    for key in example:
        return key

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
