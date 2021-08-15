# The following code has been prepared using Jupyter Notebook

# Import necessary libraries
import numpy as np
import pandas as pd
import random
import math
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

# Load the CSV file for the feature set
df = pd.read_csv('name.csv')
df=df[df.columns[1:]]
df.head()

# Segregate the features from the class labels
tot_features = len(df.columns)
print(df.shape)

x = df[df.columns[:tot_features]]
x.head()

y=[]

for i in range(df.shape[0]):
    if i<116:
        y.append(1)
    else:
        y.append(0)
        
y=pd.DataFrame(y)

y.head()
print(y.shape)

# Train random forest and MLP classifiers without implementation of BRDA
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)

# Implement BRDA

#V-shaped
def V_transfer_func(arr):    
    return np.abs(arr / np.sqrt(1+ arr*arr))
    
# Signoid function
def S_transfer_func(arr):    
    return np.exp(arr)/(1+np.exp(arr))
    
alpha = 0.2

# Define the fitness function
def find_fitness(arr):
        arr = S_transfer_func(arr)
        mask = np.where(arr>=0.5, True, False)
        #features = df.columns[:tot_features][mask]
        features = []
        for i in range(tot_features):
            if mask[i] == True:
                features.append(df.columns[i])
        if(len(features)==0):
            return 10000
        
        new_x_train = x_train[features].copy()
        new_x_test = x_test[features].copy()

        _classifier = GaussianNB()
        _classifier.fit(new_x_train, y_train)
        predictions = _classifier.predict(new_x_test)
        
        acc = accuracy_score(y_true=y_test, y_pred=predictions)
        err = 1-acc
        num_features = len(features)
        fitness = alpha*(num_features/tot_features) + (1-alpha)*(err)
        
        ## Fitness is calculated by the formula: Fitness = alpha*(No. of features in current particle / Total no. of features)
        ##                                          + (1-alpha)*(Error of current particle)

        #print("Fitness ",fitness)
        
        return fitness
        
pop_size = 260
n_male = 20
n_hind = pop_size - n_male

# The red deer population is defined
pop = np.random.uniform(low=-1, high=1, size=(pop_size, tot_features))
fitnesses = np.zeros(pop_size)
for i in range(pop_size):
    fitnesses[i] = find_fitness(pop[i])
sortidx = np.argsort(fitnesses)
pop = pop[sortidx]

best_deer = pop[0]
best_fitness = np.min(fitnesses)

max_iter = 20
UB = 1
LB = -1
gamma = 0.5
alpha = 0.2
beta = 0.1

for iter in range(max_iter):
    males = np.random.uniform(size=(n_male, tot_features))
    hinds = np.random.uniform(size=(n_hind, tot_features))
    for i in range(pop_size):
        if i<n_male:
            males[i] = pop[i].copy()
        else:
            hinds[i-n_male] = pop[i].copy()
    
    for i in range(n_male):
        a1 = random.random()
        a2 = random.random()
        a3 = random.random()
        tmp = males[i].copy()
        if a3>=0.5:
            tmp += a1*(((UB-LB)*a2)+LB)
        else:
            tmp -= a1*(((UB-LB)*a2)+LB)
        if find_fitness(tmp) < find_fitness(males[i]):
            males[i] = tmp.copy()
    
    fitnesses = np.zeros(n_male)
    for i in range(n_male):
        fitnesses[i] = find_fitness(males[i])
    sortidx = np.argsort(fitnesses)
    males = males[sortidx]
    
    # Determination of commanders and stags
    n_com = int(n_male * gamma)
    n_stag = n_male - n_com
    
    coms = np.random.uniform(size=(n_com, tot_features))
    stags = np.random.uniform(size=(n_stag, tot_features))
    for i in range(n_male):
        if i<n_com:
            coms[i] = males[i].copy()
        else:
            stags[i-n_com] = males[i].copy()
           
    # Competition between stags and comanders to ensure a position resulting in maximum roaring     
    for i in range(n_com):
        x = coms[i].copy()
        y = random.choice(stags)
        b1 = random.random()
        b2 = random.random()
        new1 = (x+y)/2 + b1*(((UB-LB)*b2)+LB)
        new2 = (x+y)/2 - b1*(((UB-LB)*b2)+LB)
        fitnesses = np.zeros(4)
        fitnesses[0] = find_fitness(x)
        fitnesses[1] = find_fitness(y)
        fitnesses[2] = find_fitness(new1)
        fitnesses[3] = find_fitness(new2)
        bestfit = np.min(fitnesses)
        if fitnesses[0] > fitnesses[1] and fitnesses[1]==bestfit:
            coms[i] = y.copy()
        elif fitnesses[0] > fitnesses[2] and fitnesses[2]==bestfit:
            coms[i] = new1.copy()
        elif fitnesses[0] > fitnesses[3] and fitnesses[3]==bestfit:
            coms[i] = new2.copy()   
            
    # Mating of commanders with hinds from its own harem
    fitnesses = np.zeros(n_com)
    for i in range(n_com):
        fitnesses[i] = find_fitness(coms[i])
    sortidx = np.argsort(fitnesses)
    coms = coms[sortidx]
    
    fsum = np.sum(fitnesses)
    fitness2 = np.zeros(n_com)
    for i in range(n_com):
        fitness2[i] = fsum - fitnesses[i]
    psum = np.sum(fitness2)
    pcom = np.zeros(n_com)
    for i in range(n_com):
        pcom[i] = fitness2[i]/psum
    n_harem = np.zeros((n_com), dtype=int)
    for i in range(n_com):
        n_harem[i] = n_hind*pcom[i]
    max_n_harem = np.max(n_harem)
    harem = np.random.uniform(size=(n_com, max_n_harem, tot_features))
    hindidx = {}
    for i in range(n_hind):
        hindidx[i] = i
    for i in range(n_com):
        sz = n_harem[i]
        for j in range(sz):
            xx = random.choice(list(hindidx))
            harem[i][j] = hinds[xx]
            del hindidx[xx]
        
    # # Mating of commanders with hinds from other harem
    n_harem_mate = np.zeros((n_com), dtype=int)
    for i in range(n_com):
        n_harem_mate[i] = n_harem[i]*alpha
    pool = []
    for i in range(n_com):
        random.shuffle(harem[i])
        for j in range(n_harem_mate[i]):
            offspring = (coms[i] + harem[i][j])/2 + (UB-LB)*random.random()
            pool.append(list(offspring))
            
        k = i
        while k==i:
            k = random.choice(range(n_com))
        num_mate = int(n_harem[k]*beta)    
        random.shuffle(harem[k])
        for j in range(num_mate):
            offspring = (coms[i] + harem[k][j])/2 + (UB-LB)*random.random()
            pool.append(list(offspring))   
    
    # # Mating of stags
    for stag in stags:
        dist = np.zeros(n_hind)
        for i in range(n_hind):
            dist[i] = math.sqrt(np.sum((stag-hinds[i])*(stag-hinds[i])))
        mindist = np.min(dist)
        for i in range(n_hind):
            distance = math.sqrt(np.sum((stag-hinds[i])*(stag-hinds[i])))
            if(distance == mindist):
                offspring = (stag + hinds[i])/2 + (UB-LB)*random.random()
                pool.append(list(offspring))
                break
                
    # Determination of the next generation
    for i in range(pop_size):
        pool.append(list(pop[i]))
    fitnesses = np.zeros(len(pool))
#     for i in range(n_male):
#         fitnesses[i] = find_fitness(males[i])
#     sortidx = np.argsort(fitnesses)
#     males = males[sortidx]    
#     fitnesses = []
    for i in range(len(pool)):
        fitnesses[i] = (find_fitness(pool[i]))
    sortidx = np.argsort(fitnesses)
#     print(sortidx)
#     print(pool)
    for i in range(pop_size):
        pop[i] = pool[sortidx[i]]
    males = np.random.uniform(size=(n_male, tot_features))
    hinds = np.random.uniform(size=(n_hind, tot_features))
    for i in range(pop_size):
        if i<n_male:
            males[i] = pop[i].copy()
        else:
            hinds[i-n_male] = pop[i].copy()    
            
best_deer = pop[0]
selected_features = np.where(S_transfer_func(best_deer)>0.5, 1, 0)
selected_features

# features = df.columns[:tot_features][selected_features]
features = []
for i in range(tot_features):
    if selected_features[i] == 1:
        features.append(df.columns[i])
#print(features)
new_x_train = x_train[features].copy()
new_x_test = x_test[features].copy()

rf1 = RandomForestClassifier(max_depth=2, random_state=0)
rf1.fit(x_train, y_train)
predictions_rf1 = rf1.predict(x_test)
total_acc_rf1 = accuracy_score(y_true = y_test, y_pred = predictions_rf1)
total_error_rf1 = 1 - total_acc_rf1

print("Random Forest")
print("Accuracy\t{}\nError\t\t{}".format(total_acc_rf1, total_error_rf1))

mlp1 = MLPClassifier(random_state=1, max_iter=300).fit(x_train, y_train)
#mlp.fit(x_train, y_train)
predictions_mlp1 = mlp1.predict(x_test)
total_acc_mlp1 = accuracy_score(y_true = y_test, y_pred = predictions_mlp1)
total_error_mlp1 = 1 - total_acc_mlp1

print("MLP")
print("Accuracy\t{}\nError\t\t{}".format(total_acc_mlp1, total_error_mlp1))
