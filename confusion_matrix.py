#CONFUSION MATRIX

num_class = 15
def make_confusion_matrix(model, x_val, y_val) :
    cols = ['Predicted "' + y  + '"' for y in one_hot.keys()]
    rows = ['Expected "' + y  + '"' for y in one_hot.keys()]
    print('CONFUSION MATRIX')
    print(np.array(list(range(15))))
    matrix =  [[0 for x in range(num_class)] for y in range(num_class)]
    #rows are expected, cols are predicted
    n = len(x_val)
    for i in range(n) :
        tmp = [x_val[i]]
        tmp = np.array(tmp)
#         y_val[i] = np.array([y_val[i]])
        exp = np.argmax(y_val[i])
        pre = np.argmax(model.predict(tmp))
        matrix[exp][pre] += 1

    # for i in range(num_class) :
    #     print(i, matrix[i], sep = "\t")
    # #print('\t\t', list(one_hot.keys()))
    print(np.matrix(matrix))
    metrics = [[0 for x in range(3)] for y in range(15)]
    #print(np.matrix(metrics))
    k = []
    for i in range(num_class) :
        metrics[i][0] = "{0:.4f}".format(get_accuracy(matrix, i))
    #print('ACCURACY : ', k)
    
    k = []
    for i in range(num_class) :
        metrics[i][1] = "{0:.4f}".format(get_precision(matrix, i))
    #print('PRECISION : ', k)
    
    k = []
    for i in range(num_class) :
        metrics[i][2] = "{0:.4f}".format(get_recall(matrix, i))
    #print('RECALL : ', k)
    print(inv_hot)
    q = ['ACCURACY', 'PRECISION', 'RECALL']
    metrics.insert(0, q)
    print(np.matrix(metrics))

def get_accuracy(matrix, item) :
    correct = matrix[item][item]
    total = 0
    for i in range(num_class) :
        total += matrix[i][item]
    if total == 0 :
        return 0
    return correct/total
def  get_recall(matrix, item):
    correct = matrix[item][item]
    total = 0
    for i in range(num_class) :
        total += matrix[i][i]
    if total == 0 :
        return 0
    return correct /  total

def get_precision(matrix, item) :
    correct = matrix[item][item]
    total = sum(matrix[item])
    if total == 0 :
        return 0
    return correct/total

'''
accuracy = correct predictions / total expected A
recall = correct predictions / (all correct predictions i.e. diagonals)  TP / (TP + FN)
precision = correct predictions / total predictions  TP / (TP + FP)
'''
from keras.models import load_model

c_model = load_model('transfer_learning_model.h5')
make_confusion_matrix(c_model, x_test, z_test)