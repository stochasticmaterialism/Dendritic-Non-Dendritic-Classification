import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets

from Py_FS.wrapper.nature_inspired._utilities import Solution, Data, initialize, sort_agents, display, compute_fitness, Conv_plot
from Py_FS.wrapper.nature_inspired._transfer_functions import get_trans_function


def GWO(num_agents, max_iter, train_data, train_label, obj_function=compute_fitness, trans_func_shape='s', save_conv_graph=False):
    
    # Grey Wolf Optimizer
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of greywolves                                          #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #                
    #   obj_function: the function to maximize while doing feature selection      #
    #   trans_function_shape: shape of the transfer function used                 #
    #   save_conv_graph: boolean value for saving convergence graph               #
    #                                                                             #
    ###############################################################################
    
    short_name = 'GWO'
    agent_name = 'Greywolf'
    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]
    trans_function = get_trans_function(trans_func_shape)

    # setting up the objectives
    weight_acc = None
    if(obj_function==compute_fitness):
        weight_acc = float(input('Weight for the classification accuracy [0-1]: '))
    obj = (obj_function, weight_acc)
    compute_accuracy = (compute_fitness, 1) # compute_accuracy is just compute_fitness with accuracy weight as 1

    # initialize greywolves and Leader (the agent with the max fitness)
    greywolves = initialize(num_agents, num_features)
    fitness = np.zeros(num_agents)
    accuracy = np.zeros(num_agents)
    Leader_agent = np.zeros((1, num_features))
    Leader_fitness = float("-inf")
    Leader_accuracy = float("-inf")

    # initialize convergence curves
    convergence_curve = {}
    convergence_curve['fitness'] = np.zeros(max_iter)

    # initialize data class
    data = Data()
    val_size = float(input('Enter the percentage of data wanted for valdiation [0, 100]: '))/100
    data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(train_data, train_label, stratify=train_label, test_size=val_size)
    
    # create a solution object
    solution = Solution()
    solution.num_agents = num_agents
    solution.max_iter = max_iter
    solution.num_features = num_features
    solution.obj_function = obj_function

    # rank initial greywolves
    greywolves, fitness = sort_agents(greywolves, obj, data)

    # start timer
    start_time = time.time()

    # initialize the alpha, beta and delta grey wolves and their fitness
    alpha, beta, delta = np.zeros((1, num_features)), np.zeros((1, num_features)), np.zeros((1, num_features))
    alpha_fit, beta_fit, delta_fit = float("-inf"), float("-inf"), float("-inf")

    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no+1))
        print('================================================================================\n')

        # update the alpha, beta and delta grey wolves
        for i in range(num_agents):

            # update alpha, beta, delta
            if fitness[i] > alpha_fit:
                delta_fit = beta_fit
                delta = beta.copy()
                beta_fit = alpha_fit
                beta = alpha.copy()
                alpha_fit = fitness[i]
                alpha = greywolves[i, :].copy()

            # update beta, delta
            elif fitness[i] > beta_fit:
                delta_fit = beta_fit
                delta = beta.copy()
                beta_fit = fitness[i]
                beta = greywolves[i, :].copy()

            # update delta
            elif fitness[i] > delta_fit:
                delta_fit = fitness[i]
                delta = greywolves[i, :].copy()

        # a decreases linearly fron 2 to 0
        a = 2-iter_no*((2)/max_iter)

        for i in range(num_agents):
            for j in range(num_features):  

                # calculate distance between alpha and current agent
                r1 = np.random.random() # r1 is a random number in [0,1]
                r2 = np.random.random() # r2 is a random number in [0,1]
                A1 = (2 * a * r1) - a # calculate A1 
                C1 = 2 * r2 # calculate C1
                D_alpha = abs(C1 * alpha[j] - greywolves[i, j]) # find distance from alpha
                X1 = alpha[j] - (A1 * D_alpha) # Eq. (3.6)

                # calculate distance between beta and current agent
                r1 = np.random.random() # r1 is a random number in [0,1]
                r2 = np.random.random() # r2 is a random number in [0,1]
                A2 = (2 * a * r1) - a # calculate A2
                C2 = 2 * r2 # calculate C2
                D_beta = abs(C2 * beta[j] - greywolves[i, j]) # find distance from beta
                X2 = beta[j] - (A2 * D_beta) # Eq. (3.6)

                # calculate distance between delta and current agent
                r1 = np.random.random() # r1 is a random number in [0,1]
                r2 = np.random.random() # r2 is a random number in [0,1]
                A3 = (2* a * r1) - a # calculate A3
                C3 = 2 * r2 # calculate C3
                D_delta = abs(C3 * delta[j] - greywolves[i, j]) # find distance from delta
                X3 = delta[j]-A3*D_delta # Eq. (3.6)

                # update the position of current agent
                greywolves[i, j] = (X1 + X2 + X3) / 3 # Eq. (3.7)

            # Apply transformation function on the updated greywolf
            for j in range(num_features):
                trans_value = trans_function(greywolves[i,j])
                if (np.random.random() < trans_value): 
                    greywolves[i,j] = 1
                else:
                    greywolves[i,j] = 0

        # update final information
        greywolves, fitness = sort_agents(greywolves, obj, data)
        display(greywolves, fitness, agent_name)
        
        # update Leader (best agent)
        if fitness[0] > Leader_fitness:
            Leader_agent = greywolves[0].copy()
            Leader_fitness = fitness[0].copy()

        if alpha_fit > Leader_fitness:
            Leader_fitness = alpha_fit
            Leader_agent = alpha.copy()


        convergence_curve['fitness'][iter_no] = np.mean(fitness)


    # compute final accuracy
    Leader_agent, Leader_accuracy = sort_agents(Leader_agent, compute_accuracy, data)
    greywolves, accuracy = sort_agents(greywolves, compute_accuracy, data)

    print('\n================================================================================')
    print('                                    Final Result                                  ')
    print('================================================================================\n')
    print('Leader ' + agent_name + ' Dimension : {}'.format(int(np.sum(Leader_agent))))
    print('Leader ' + agent_name + ' Fitness : {}'.format(Leader_fitness))
    print('Leader ' + agent_name + ' Classification Accuracy : {}'.format(Leader_accuracy))
    print('\n================================================================================\n')

    # stop timer
    end_time = time.time()
    exec_time = end_time - start_time

    # plot convergence graph
    fig, axes = Conv_plot(convergence_curve)
    if(save_conv_graph):
        plt.savefig('convergence_graph_'+ short_name + '.jpg')
    plt.show()

    # update attributes of solution
    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.best_accuracy = Leader_accuracy
    solution.convergence_curve = convergence_curve
    solution.final_greywolves = greywolves
    solution.final_fitness = fitness
    solution.final_accuracy = accuracy
    solution.execution_time = exec_time

    return solution


if __name__ == '__main__':

    data = datasets.load_digits()
    GWO(20, 100, data.data, data.target, save_conv_graph=True)
