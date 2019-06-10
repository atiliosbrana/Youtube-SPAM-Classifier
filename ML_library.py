import numpy as np

#K-Nearest Neighbors
def normalizar(df):
    df = pd.DataFrame(df)
    X_norm = (df - df.mean()) / df.std()
    mu = df.mean().values
    sigma = df.std().values
    return X_norm.values, mu, sigma

def knn_distancia(x, X):
    D = np.linalg.norm(x-X)
    return D

def knn(x, X, Y, K):
    
    D = []
    for i in X:
        D.append(knn_distancia(x, i))
    
    D_dict = {}
    for idx, val in enumerate(D):
        D_dict[idx] = val
    #indices dos K valores
    index_K = sorted(D_dict, key=D_dict.get, reverse=False)[:K]
    #classes dos indices dos K valores
    classes_K = list(Y[index_K])
    y_pred = max(set(classes_K), key=classes_K.count)
    
    return y_pred

#Regressão Logística
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def funcaoCusto(theta, X, Y):
    m = len(Y)
    eps = 1e-15
    h = sigmoid(X.dot(theta))
    J = -(1 / m) * np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h + eps))
    grad = (1 / m) * np.dot(X.T, h - Y)
    return J, grad

def predicao(theta, X):
    return np.round(sigmoid(X.dot(theta)) + 1e-15)

def funcaoCustoReg(theta, X, Y, l):
    m = len(Y)
    eps = 1e-15
    h = sigmoid(X.dot(theta))
    J = -(1 / m) * np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h + eps)) + (l / (2 * m)) * np.sum(np.square(theta[1:]))
    reg = l / m * theta[1:]
    grad = (1 / m) * np.dot(X.T, h - Y)
    grad[1:] += reg
    return J, grad

#Naive Bayes
def calcularProbabilidades(X, Y):
    p1, p0 = np.sum(X[Y == 1], axis = 0) / len(X[Y == 1]), np.sum(X[Y == 0], axis = 0) / len(X[Y == 0])
    return p1, p0

def classificacao(x, p1, p0, pAtr1, pAtr0):
    prob1 = p1 * np.prod([pAtr1[i] if x[i] else (1 - pAtr1[i]) for i in range(len(x))])
    prob0 = p0 * np.prod([pAtr0[i] if x[i] else (1 - pAtr0[i]) for i in range(len(x))])
    classe = 1 if prob1 > prob0 else 0
    return classe, prob1, prob0

#Metodologia

def stratified_holdOut(target, pTrain):
    from math import ceil
    classes = set(target)
    n = len(target)
    train_index = list()
    test_index = list()
    for nclass in classes:
        class_index = [x for x, y in list(enumerate(target)) if y == nclass]
        n = len(class_index)
        cut = ceil(n * pTrain)
        train_index.extend(class_index[:cut])
        test_index.extend(class_index[cut:])
    train_index = np.array(sorted(train_index))
    test_index = np.array(sorted(test_index))
    return train_index, test_index

def get_confusionMatrix(Y_test, Y_pred, classes):
    cm = np.zeros( [len(classes),len(classes)], dtype=int )
    for test, pred in zip(np.array(Y_test, dtype = int), np.array(Y_pred, dtype = int)):
        cm[test, pred] += 1 
    return cm

def curva_aprendizado_knn(X, Y, Xval, Yval, K):

    # inicializa as listas que guardarao a performance no treinamento e na validacao
    perf_train = []
    perf_val = []
    
    num_rows = len(X)
    
    for i in range(100,num_rows, 500):
        print(i)
        x_train = X[0:i,:]
        y_train = Y[0:i]
        
        #pred
        #train
        pred_train = []
        for sample in x_train:
            y_pred_sample = ML_library.knn(sample, x_train, y_train, K)
            pred_train.append(y_pred_sample)            
        #validation
        pred_val = []
        for sample in Xval:
            y_pred_sample = ML_library.knn(sample, Xval, Yval, K)
            pred_val.append(y_pred_sample)          
        
        acc_train = len([i for i, j in zip(pred_train, y_train) if i == j])/len(pred_train)
        acc_val = len([i for i, j in zip(pred_val, Yval) if i == j])/len(pred_val)
        
        perf_train.append(acc_train)
        perf_val.append(acc_val)
       
    # Define o tamanho da figura 
    plt.figure(figsize=(10,6))

    # Plota os dados
    plt.plot(perf_train, color='blue', linestyle='-', linewidth=1.5, label='Treino') 
    plt.plot(perf_val, color='red', linestyle='-', linewidth=1.5, label='Validação')

    # Define os nomes do eixo x e do eixo y
    plt.xlabel(r'# Qtd. de dados de treinamento',fontsize='x-large') 
    plt.ylabel(r'Score',fontsize='x-large') 

    # Define o título do gráfico
    plt.title(r'Curva de aprendizado', fontsize='x-large')

    # Acrescenta um grid no gráfico
    plt.grid(axis='both')

    # Plota a legenda
    plt.legend()
    
    plt.show()


def relatorioDesempenho(matriz_confusao, classes, imprimeRelatorio=False):
    n_teste = sum(sum(matriz_confusao))
    nClasses = len( matriz_confusao )
    vp=np.zeros( nClasses )
    vn=np.zeros( nClasses )
    fp=np.zeros( nClasses )
    fn=np.zeros( nClasses )
    revocacao = np.zeros( nClasses )
    precisao = np.zeros( nClasses )
    fmedida = np.zeros( nClasses )

    for nclass in classes:
        vp[nclass] = matriz_confusao[nclass, nclass]
        fp[nclass] = matriz_confusao[:, nclass].sum() - vp[nclass]
        fn[nclass] = matriz_confusao[nclass].sum() - vp[nclass]
        vn[nclass] = matriz_confusao.sum() - vp[nclass] - fp[nclass] - fn[nclass]
        vp_, fp_, fn_, vn_ = vp[nclass], fp[nclass], fn[nclass], vn[nclass]
        
        precisao[nclass] = vp_ / (vp_ + fp_)
        revocacao[nclass] = vp_ / (vp_ + fn_)
        p_, r_ = precisao[nclass], revocacao[nclass]
        
        fmedida[nclass] = 2 * p_ * r_ / (p_ + r_)
        
    acuracia = (vp.sum() + vn.sum()) / (vp + vn + fp + fn).sum()

    revocacao_macroAverage = np.mean(revocacao)
    revocacao_microAverage = vp.sum() / (vp + fn).sum()

    precisao_macroAverage = np.mean(precisao)
    precisao_microAverage = vp.sum() / (vp + fp).sum()

    fmedida_macroAverage = 2 * revocacao_macroAverage * precisao_macroAverage / (revocacao_macroAverage + precisao_macroAverage)
    fmedida_microAverage = 2 * revocacao_microAverage * precisao_microAverage / (revocacao_microAverage + precisao_microAverage)

    if imprimeRelatorio:
      print('\n\tRevocacao   Precisao   F-medida   Classe')
      for i in range(0,nClasses):
        print('\t%1.3f       %1.3f      %1.3f      %s' % (revocacao[i], precisao[i], fmedida[i],classes[i] ) )

      print('\t------------------------------------------------');

      #imprime as médias
      print('\t%1.3f       %1.3f      %1.3f      Média macro' % (revocacao_macroAverage, precisao_macroAverage, fmedida_macroAverage) )
      print('\t%1.3f       %1.3f      %1.3f      Média micro\n' % (revocacao_microAverage, precisao_microAverage, fmedida_microAverage) )

      print('\tAcuracia: %1.3f' %acuracia)

    resultados = {'revocacao': revocacao, 'acuracia': acuracia, 'precisao':precisao, 'fmedida':fmedida}
    resultados.update({'revocacao_macroAverage':revocacao_macroAverage, 'precisao_macroAverage':precisao_macroAverage, 'fmedida_macroAverage':fmedida_macroAverage})
    resultados.update({'revocacao_microAverage':revocacao_microAverage, 'precisao_microAverage':precisao_microAverage, 'fmedida_microAverage':fmedida_microAverage})
    resultados.update({'confusionMatrix': matriz_confusao})

    return resultados

def gridSearch(X, Y, Xval, Yval):
    bestReg = -100
    reg = [0,0.5,1,10,50,100]
    
    iteracoes = 500
    f1 = -10**6
    
    for lambda_reg in reg:
        theta = treinamento(X, Y, lambda_reg, iteracoes)
        Y_pred_val = predicao(Xval, theta)
        
        cm = get_confusionMatrix(Yval, Y_pred_val, [0,1])
        vp = cm[0,0] # quantidade de verdadeiros positivos
        vn = cm[1,1] # quantidade de verdadeiros negativos
        fp = cm[1,0] # quantidade de falsos positivos
        fn = cm[0,1] # quantidade de falsos negativos
        
        recall_train = vp_train/(vp_train+fn_train)
        precision_train = vp_train/(vp_train+fp_train)
    
        f1_val = 2 * (precision_train * recall_train) / (precision_train + recall_train)
        
        if f1_val > f1:
            bestReg = lambda_reg
            f1 = f1_val
    return bestReg

def stratified_kfolds(target, k, classes):
    folds_final = np.zeros( k,dtype='object')
    train_index = np.zeros( k,dtype='object')
    test_index = np.zeros( k,dtype='object')
    for i in range(len(folds_final)):
        
        train_index[i] = np.array([], dtype = int)
        test_index[i] = np.array([], dtype = int)
        folds_final[i] = [[], []]

    from math import ceil
    for nclass in classes:
        class_index = [x for x, y in list(enumerate(target)) if y == nclass]    
        n = len(class_index)
        band = ceil(n / k)
        cut1 = 0
        for i in range(k):
            cut0 = cut1
            cut1 = cut0 + band
            test_index = class_index[cut0:cut1]
            train_index = class_index[:cut0] + class_index[cut1:]
            folds_final[i][0] = np.array(np.append(folds_final[i][0], train_index), dtype = int)
            folds_final[i][1] = np.array(np.append(folds_final[i][1], test_index), dtype = int)

    return folds_final

#Neural Networks
def funcaoCusto(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y):
    Theta1 = np.reshape( nn_params[0:hidden_layer_size*(input_layer_size + 1)], (hidden_layer_size, input_layer_size+1) )
    Theta2 = np.reshape( nn_params[ hidden_layer_size*(input_layer_size + 1):], (num_labels, hidden_layer_size+1) )
    m = X.shape[0]
    Y = np.eye(num_labels)[y - 1]
    a1 = np.append(np.ones((m, 1)), X, axis = 1)
    a2 = np.append(np.ones((m, 1)), sigmoid(np.dot(a1, Theta1.T)), axis = 1)
    h = sigmoid(np.dot(a2, Theta2.T))
    J = - (1 / m) * np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h))
    return J

def funcaoCusto_reg(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, vLambda):
    Theta1 = np.reshape( nn_params[0:hidden_layer_size*(input_layer_size + 1)], (hidden_layer_size, input_layer_size+1) )
    Theta2 = np.reshape( nn_params[ hidden_layer_size*(input_layer_size + 1):], (num_labels, hidden_layer_size+1) )
    m = X.shape[0]
    J = funcaoCusto(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y)
    T1 = np.sum(Theta1[:,1:] ** 2)
    T2 = np.sum(Theta2[:,1:] ** 2)
    J = J + (vLambda / (2 * m) * (T1 + T2))
    return J

def inicializaPesosAleatorios(L_in, L_out, randomSeed = None):
    epsilon_init = 0.12
    if randomSeed is not None:
        W = np.random.RandomState(randomSeed).rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    else:
        W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init     
    return W

def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def funcaoCusto_backp(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y):
    Theta1 = np.reshape( nn_params[0:hidden_layer_size*(input_layer_size + 1)], (hidden_layer_size, input_layer_size+1) )
    Theta2 = np.reshape( nn_params[ hidden_layer_size*(input_layer_size + 1):], (num_labels, hidden_layer_size+1) )

    m = X.shape[0]
         
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    Y = np.eye(num_labels)[y - 1]
    a1 = np.append(np.ones((m, 1)), X, axis = 1)
    a2 = np.append(np.ones((m, 1)), sigmoid(np.dot(a1, Theta1.T)), axis = 1)
    h = sigmoid(np.dot(a2, Theta2.T))
    J = - (1 / m) * np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h))
    
    D1, D2 = 0, 0
    for i in range(m):
        x = X[i]
        a1 = np.append([1], x)
        a2 = np.append([1], sigmoid(np.dot(a1, Theta1.T)))
        a3 = sigmoid(np.dot(a2, Theta2.T))
        d3 = a3 - Y[i]
        d2 = np.dot(Theta2[:,1:].T, d3).T * sigmoidGradient(np.dot(a1, Theta1.T))
        D1 += np.outer(d2, a1)
        D2 += np.outer(d3, a2)
    
    Theta1_grad = D1 / m
    Theta2_grad = D2 / m

    grad = np.concatenate([np.ravel(Theta1_grad), np.ravel(Theta2_grad)])

    return J, grad

def funcaoCusto_backp_reg(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, vLambda):
    Theta1 = np.reshape( nn_params[0:hidden_layer_size*(input_layer_size + 1)], (hidden_layer_size, input_layer_size+1) )
    Theta2 = np.reshape( nn_params[ hidden_layer_size*(input_layer_size + 1):], (num_labels, hidden_layer_size+1) )

    m = X.shape[0]
         
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    Y = np.eye(num_labels)[y - 1]
    a1 = np.append(np.ones((m, 1)), X, axis = 1)
    a2 = np.append(np.ones((m, 1)), sigmoid(np.dot(a1, Theta1.T)), axis = 1)
    h = sigmoid(np.dot(a2, Theta2.T))
    J = - (1 / m) * np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h))
    
    T1 = np.sum(Theta1[:,1:] ** 2)
    T2 = np.sum(Theta2[:,1:] ** 2)

    J = J + (vLambda / (2 * m) * (T1 + T2))
    
    D1, D2 = 0, 0
    for i in range(m):
        x = X[i]
        a1 = np.append([1], x)
        a2 = np.append([1], sigmoid(np.dot(a1, Theta1.T)))
        a3 = sigmoid(np.dot(a2, Theta2.T))
        d3 = a3 - Y[i]
        d2 = np.dot(Theta2[:,1:].T, d3).T * sigmoidGradient(np.dot(a1, Theta1.T))
        D1 += np.outer(d2, a1)
        D2 += np.outer(d3, a2)
    
    Theta1_grad = D1 / m
    Theta2_grad = D2 / m
    
    Theta1_1st_j = np.copy(Theta1_grad[:,0])
    Theta2_1st_j = np.copy(Theta2_grad[:,0])
    Theta1_grad += vLambda / m * Theta1
    Theta2_grad += vLambda / m * Theta2
    Theta1_grad[:,0] = Theta1_1st_j
    Theta2_grad[:,0] = Theta2_1st_j
    
    grad = np.concatenate([np.ravel(Theta1_grad), np.ravel(Theta2_grad)])

    return J, grad

#SVM
def gridSearch(X, Y, Xval, Yval):
    custo = 1000
    gamma = 1000
    v = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    acuracia_max = -1
    for i in v:
        for j in v:
            temp_custo, temp_gamma = i, j
            model = svm_train(Y2, X2, '-c %f -t %d -g %f' %(temp_custo, kernel, temp_gamma))
            classes = svm_predict(Yval, Xval, model)
            acuracia = np.mean(classes[0]==Yval)
            if acuracia > acuracia_max:
                custo, gamma, acuracia_max = temp_custo, temp_gamma, acuracia
    return custo, gamma

#K-mean
def findClosestCentroids(X, centroids):
    n = X.shape[0]
    K = centroids.shape[0]
    idx = np.zeros( X.shape[0], dtype=int ) 

    for i in range(n):
        for j in range(K):
            d = X[i] - centroids[j]
            dist = np.sqrt(np.dot(d, d))
            if j == 0 or dist < min_dist:
                min_dist, cent = dist, j
        idx[i] = cent
        
    return idx

def calculateCentroids(X, idx, K):
    return np.array([np.mean(X[idx == i], axis = 0) for i in range(K)])

def executa_kmedias(X, initial_centroids, max_iters, saveHistory = False):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros( m ) 
    if saveHistory: history = [] 
    for i in range(max_iters):
        print('K-Médias - Iteração %d/%d' %(i, max_iters));
        idx = findClosestCentroids(X, centroids)
        if saveHistory:
            history.append({'centroids': centroids, 'idx': idx})
        centroids = calculateCentroids(X, idx, K);
    if saveHistory:
        return centroids, idx, history
    else:
        return centroids, idx
