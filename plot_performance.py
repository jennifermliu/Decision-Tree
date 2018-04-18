import ID3, parse, random
import matplotlib.pyplot as plt
import numpy as np

# inFile - string location of the house data file
def testPruningOnHouseData(inFile):
  withPruningAvg = []
  withoutPruningAvg = []
  data = parse.parse(inFile)
  error = 0
  worsecase = 0
  for k in range(10,305,5):
    withPruning = []
    withoutPruning = []
    for i in range(100):
      random.shuffle(data)
      train = data[:k*7/10]
      valid = data[k*7/10:k]
      test = data[k:]

      tree = ID3.ID3(train, 'democrat')
      acc = ID3.test(tree, train)
      # print "training accuracy: ",acc
      acc = ID3.test(tree, valid)
      # validOrig = acc
      # print "validation accuracy: ",acc
      acc = ID3.test(tree, test)
      # print "test accuracy: ",acc

      ID3.prune(tree, valid)
      acc = ID3.test(tree, train)
      # print "pruned tree train accuracy: ",acc
      acc = ID3.test(tree, valid)
      # validPrun = acc
      # if validPrun < validOrig:
      #   error += 1
      # print "pruned tree validation accuracy: ",acc
      acc = ID3.test(tree, test)
      # print "pruned tree test accuracy: ",acc
      # testOrig = acc
      withPruning.append(acc)
      tree = ID3.ID3(train+valid, 'democrat')
      acc = ID3.test(tree, test)
      # testPruning = acc
      # if testPruning < testOrig:
      #   worsecase += 1
      # print "no pruning test accuracy: ",acc
      withoutPruning.append(acc)
    # print withPruning
    # print withoutPruning
    # print len(withPruning)
    withPruningAvg.append(sum(withPruning)/len(withPruning))
    withoutPruningAvg.append(sum(withoutPruning)/len(withoutPruning))


  fig = plt.figure()
  #ax = plt.axes()
  x = range(10,305,5)
  # y1 = range(10,300,10)
  # y2 = range(10,155,5)
  y1 = withoutPruningAvg
  y2 = withPruningAvg

  #x = np.linspace(0, 300, 1)
  plt.plot(x, y1, label = 'fitting plot without pruning',color='red')
  plt.scatter(x, y1, label= "scatter plot without pruning", color= "red", marker= "*")
  plt.plot(x, y2, label = 'fitting plot with pruning', color='blue')
  plt.scatter(x, y2, label= "scatter plot with pruning", color= "blue", marker= "*")
  plt.xlabel('number of training examples')
  # naming the y axis
  plt.ylabel('average accuracy on test data')
 
  # giving a title to my graph
  plt.title('comparision between non-pruning tree and pruning tree')
  plt.legend()
  plt.show()
  
    # print "average with pruning",sum(withPruning)/len(withPruning)," without: ",sum(withoutPruning)/len(withoutPruning)
    # print error
    # print worsecase

testPruningOnHouseData("house_votes_84.data")
