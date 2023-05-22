import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy


def fitness(number):
    #return "{0:04b}".format(number).count("1")
    return number**2


# plotting the fitness landscape of the int_bin problem
#a = sns.lineplot(data=[fitness(i) for i in range(0, 16)])
#a.set_xticks(range(0, 16))
#plt.show()


# new neighbourhood structure
rep = [0, 0, 0, 0]
n = [deepcopy(rep) for i in range(len(rep))]

for count, i in enumerate(n):
    if i[count] == 1:
        i[count] = 0
    elif i[count] == 0:
        i[count] = 1

#print(n)
#print(rep)


# simulated annealing parameters
def plot_c(c, alpha, threshold):
    c_list = [c]
    while c > threshold:
        c = c * alpha
        c_list.append(c)
    plt.plot(c_list)
    plt.show()


plot_c(10, 0.95, 0.05)


