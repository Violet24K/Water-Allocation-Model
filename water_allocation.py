from numpy.random import randint
from numpy.random import rand
import math
import numpy

# global variables
# Given: P # initial water level of Lake Powell
# Given: M # initial water level of Lake Mead
# calculated: # water for direct general-usage from Dam i
# calculated: E_i # water for electric production from Dam i
# Given: n_j # water demand of state j
# calculated: n_a_j#additional water demand for state j
# s_j # additional water supply for state j
# s_j_s # amount of static form additional water supply for state j (e.g. river_ lake_ underground water_ etc.)
# s_j_d # average amount of dynamic form additional water supply for state j (e.g. rainfall).
# epsilon_j# proportion of static form additional water for state j
# di,j Euclidean distance between dam i to state j
# !t_i_j # water delivery time from dam i to state j
# Assigned: alpha # water loss rate from dam to state_ considering all possible causes
# Assigned: beta # electricity loss rate during transportation
# assigned: W_j # total water consumed for state j in the previous time period


# Q_j # total electricity consumed by state j in the previous time period
# L_j # loss from water for state j
# B_j # profit for state j in the last time period
# calculated b_w_j # average water benefit for state j
# calculated b_j # average electrical benefit for state j
# needs calculated: q_w_i_j # the amount of water drawn from dam i to state j
# actually not used: q_e_i_j # the amount of electricity drawn from dam i to state j


# looked up: H_f # maximum height of dam when the dam is full
# made up: H_d # height of water at the stop line of electricity generation
# lokked up: A_f # area of transverse section at H_f
# made up: A_d # area of transverse section at H_d
# h # current height of water in the dam
# S # area of current upper surface of water in the dam
# S_c # cross sectional area of the drain window of Glen Canyon Dam
# S_d # cross sectional area of the delivery canals/pipes
# h_0 # height between dead surface and the tip of cone completed by the frustum
# v # water output rate
# #v_d # water delivery rate
# v_out # water output rate of Glen Canyon Dam
# t_P,j # water delivery time from Glen Canyon Dam to states
# t_M,j # water delivery time from Hoover Dam to states
# eta # efficiency of hydroelectric power production
# eta_1 # fraction of water for general use that flows from Lake Powell to Mead
# eta_2 # fraction of water for general use of Lake Mead
# sigma # fraction of water for hydro-electricity generation of Lake Powell
# rho # density of water
# r # radius of earth
# g # acceleration of gravity

# parameters
alpha = 0.01
beta = 0
eta = 0.9
lambdaa = 0.7  # random guess
mu = 0.05
eta_1 = 0.01

mexico_coef = 1.0

# constraint
# all demand: 234520000

total_constraint = 234520000

# water demand
n = [23870000, 114530000, 32850000, 11760000, 41510000, 10000000]
electricity_limit0 = 1/10*(n[0]+n[1]+n[2]+n[3]+n[4]+n[5])
electricity_limit1 = 1/10*(n[0]+n[1]+n[2]+n[3]+n[4]+n[5])

# height
P = 70
M = 100

# additional water supply
s = [0, 0, 0, 0, 0]

# water consumption/withdrawl per day in year 2019 (in m^3)
W = [22640000, 109020000, 30810000, 10980000, 38990000]
# Distance[dam][state] (in m)
d = [[259288, 977765, 486080, 594458, 652504],
     [334919, 620631, 838904, 951639, 1016293]]
# density of water
rho = 1000
# radius of earth (in m)
r = 6378137
# acceleration of gravity
g = 9.81
# water delivery rate (m/s)
v_d = 5
# cross section area of dam(m^2)
S_c = [18, 18]
S_d = 200
# Characteristic of Lakes
# Depth
H_f = [90, 80]
H_d = [10, 10]
# Surface
A_f = [658000000, 637050000]
# bed
A_d = [151388416, 386948241]
# total electricity consumed by state j in the previous year(megawatthours)
# (1=AZ, 2=CA, 3=WY, 4=NM,5=CO)
Q_j = [220728, 12545279, 1504219, 162685, 83313]
# profit(GDP) for state j in the last time period
B = [1130307.9, 9429811.7, 161672.6, 358805.2, 1249638.6]


# time_out
# Input: h1 & h2 - water level before and after water output
#		 dam_select - which dam to calculate
#		 eta_1 - fraction of water for general use that flows from Lake Powell to Mead
# Output: tout - time needed to output (in s)
def time_out(h1, h2, dam_select, eta_1):
    H_flood = H_f[dam_select]
    H_dead = H_d[dam_select]
    A_dead = A_d[dam_select]
    A_flood = A_f[dam_select]
    H = H_flood - H_dead
    h0 = H/(math.sqrt(A_flood/A_dead)-1)
    temp1 = 2/5*h1**(5/2) + 4/3*h0*h1**(3/2) + 2*h0**2*h1**(1/2)
    temp2 = 2/5*h2**(5/2) + 4/3*h0*h2**(3/2) + 2*h0**2*h2**(1/2)
    tout = abs(temp1-temp2)*eta_1*A_dead/S_c[dam_select]/h0**2/math.sqrt(2*g)
    return tout


def time_powell(D, state):
    return D/S_d/v_d + d[0][state]/v_d


def time_mead(D, I, h1, h2, eta_1, eta_2, q_w, n, state):
    time1 = D/S_d/v_d + d[1][state]/v_d
    time2 = eta_2*I/S_d/v_d + d[1][state]/v_d + time_out(h1, h2, 0, eta_1)
    return max(time1, time2)

# Gini coefficient
# Input: q_w[2][6] - the amount of water drawn from dam i to state j/Mexico
#		 n - the water demand of each state together with Mexico
#		 mexico_coef - the control weight of Mexico
# Output: gini (the Gini Coefficient)


def Gini(q_w, n):
    diff_sum = 0
    for l in range(6):
        for k in range(6):
            if (l == 5 or k == 5):
                diff_sum += mexico_coef * \
                    abs((q_w[0][l]+q_w[1][l])/n[l]-(q_w[0][k]+q_w[1][k])/n[k])
            else:
                diff_sum += abs((q_w[0][l]+q_w[1][l]) /
                                n[l]-(q_w[0][k]+q_w[1][k])/n[k])
    sum = 0
    for j in range(6):
        sum += (q_w[0][j]+q_w[1][j])/n[j]
    gini = 1/72/sum*diff_sum
    return gini


def ReLU(x):
    if (x >= 0):
        return x
    return 0


# 0 Glen (P), 1 Hoover (M)
def Volume(h1, h2, dam_select):
    H_flood = H_f[dam_select]
    H_dead = H_d[dam_select]
    A_dead = A_d[dam_select]
    A_flood = A_f[dam_select]
    H = H_flood - H_dead
    h0 = H/(math.sqrt(A_flood/A_dead)-1)
    temp1 = math.pow(((h1+h0)/h0), 3)
    temp2 = math.pow(((h2+h0)/h0), 3)
    V = 1/3*A_dead*h0*(temp1-temp2)
    return V


def h_second(h1, V, dam_select):
    H_flood = H_f[dam_select]
    H_dead = H_d[dam_select]
    A_dead = A_d[dam_select]
    A_flood = A_f[dam_select]
    H = H_flood - H_dead
    h0 = H/(math.sqrt(A_flood/A_dead)-1)
    h2 = math.pow(h0 * ((math.pow((h1+h0)/h0, 3)) - 3*V/(A_dead*h0) - 1), 1/3)
    return h2


def U(h1, h2, dam_select):
    H_flood = H_f[dam_select]
    H_dead = H_d[dam_select]
    A_dead = A_d[dam_select]
    A_flood = A_f[dam_select]
    H = H_flood - H_dead
    h0 = H/(math.sqrt(A_flood/A_dead)-1)
    temp1 = math.pow(h1, 4)/4 + 2*h0*math.pow(h1, 3) / \
        3 + math.pow(h0, 2)*math.pow(h1, 2)/2
    temp2 = math.pow(h2, 4)/4 + 2*h0*math.pow(h2, 3) / \
        3 + math.pow(h0, 2)*math.pow(h2, 2)/2
    result = eta * rho * g * A_dead * (temp1 - temp2) / math.pow(h0, 2)
    if (h2 < 0):
        return temp1
    else:
        return result


def Loss(q_w):
    b = [0, 0, 0, 0, 0]
    q = [0, 0, 0, 0, 0]
    L = [0, 0, 0, 0, 0]
    n_total = 0
    q_total = 0
    s_total = 0
    L_w = 0
    for j in range(5):
        b[j] = B[j] / W[j]
        q[j] = q_w[0][j] + q_w[1][j]
        L[j] = ReLU(b[j] * (n[j] - (1-alpha)*q[j] + s[j]))
        L_w += L[j]
        n_total += n[j]
        q_total += q[j]
        s_total += s[j]
        b_min = min(b[0], b[1], b[2], b[3], b[4])
        L_min = ReLU(b_min * (n_total - (1-alpha)*q_total + s_total))
        b_max = max(b[0], b[1], b[2], b[3], b[4])
        L_max = ReLU(b_max * (n_total - (1-alpha)*q_total + s_total))
        if (L_max - L_min == 0):
            L_final = 0
        else:
            L_final = (L_w - L_min)/(L_max-L_min)
    return L_final


def Benefit(E, q_w):
    q_total0 = 0
    for j in range(6):
        q_total0 += q_w[0][j]
    q_total1 = 0
    for j in range(6):
        q_total1 += q_w[1][j]
    h1_0 = P - H_d[0]
    h1_1 = M - H_d[1]
    # h2_0 = h_second(h1_0, E[0], 0)
    # h2_1 = h_second(h1_1, E[1], 1)
    h2_0_min = h_second(h1_0, E[0]+q_total0, 0)
    h2_1_min = h_second(h1_1, E[1]+q_total1, 1)
    U_1 = U(h1_0, h2_0_min, 0)*E[0]/(E[0]+q_total0)
    U_2 = U(h1_1, h2_1_min, 1)*E[1]/(E[1]+q_total1)
    # U_1_max = U(h1_0, 0, 0)
    # U_2_max = U(h1_1, 0, 1)
    U_1_max = U(h1_0, h2_0_min, 0)
    U_2_max = U(h1_1, h2_1_min, 1)
    B = (U_1+U_2)/(U_1_max + U_2_max)
    return B


def Obj(q_w, n, E):
    loss = Loss(q_w)
    benefit = Benefit(E, q_w)
    G = Gini(q_w, n)
    obj = -lambdaa*loss + mu*benefit-(1-lambdaa-mu)*G
    return obj


def constraint(x):
    q_w = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    q_w[0][0] = x[0]
    q_w[0][1] = x[1]
    q_w[0][2] = x[2]
    q_w[0][3] = x[3]
    q_w[0][4] = x[4]
    q_w[0][5] = x[5]
    q_w[1][0] = x[6]
    q_w[1][1] = x[7]
    q_w[1][2] = x[8]
    q_w[1][3] = x[9]
    q_w[1][4] = x[10]
    q_w[1][5] = x[11]
    E = [0, 0]
    E[0] = x[12]
    E[1] = x[13]
    sump = 0
    summ = 0
    VolumeP = Volume(P, 0, 0)
    VolumeM = Volume(M, 0, 1)
    Mexico_amount = q_w[0][5] + q_w[1][5]
    if (Mexico_amount < n[5]):
        return False
    for i in range(6):
        if q_w[0][i] < 0 or q_w[1][i] < 0:
            return False
        sump += q_w[0][i]
        summ += q_w[1][i]
    # if sump > 5/6*(1-alpha)*(1-eta_1)*(VolumeP):
    #     return False
    # if summ > 5/6*(1-alpha)*(eta_1*VolumeP+VolumeM):
    #     return False
    if sump + E[0] > (1-alpha)*(1-eta_1)*(VolumeP):
        return False
    if summ + E[1] > (1-alpha)*(eta_1*VolumeP+VolumeM):
        return False
    if sump + summ > (1-alpha)*(VolumeM + VolumeP):
        return False
    # for i in range(6):
    #     ratio = q_w[0][i]/q_w[1][i]
    #     if ratio < 0.35:
    #         return False
    #     if ratio > 0.65:
    #         return False

    # constraint for total water
    if (sump + summ + E[0] + E[1] > total_constraint):
        return False
    return True




# The main part of genetic algorithm

# objective function


def objective(x):
    q_w = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    q_w[0][0] = x[0]
    q_w[0][1] = x[1]
    q_w[0][2] = x[2]
    q_w[0][3] = x[3]
    q_w[0][4] = x[4]
    q_w[0][5] = x[5]
    q_w[1][0] = x[6]
    q_w[1][1] = x[7]
    q_w[1][2] = x[8]
    q_w[1][3] = x[9]
    q_w[1][4] = x[10]
    q_w[1][5] = x[11]
    E = [0, 0]
    E[0] = x[12]
    E[1] = x[13]
    y = 3 - Obj(q_w, n, E)
    return y




# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
    decoded = list()
    largest = 2**n_bits
    for i in range(len(bounds)):
        # extract the substring
        start, end = i * n_bits, (i * n_bits)+n_bits
        substring = bitstring[start:end]
        # convert bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range
        value = bounds[i][0] + (integer/largest) * \
            (bounds[i][1] - bounds[i][0])
        # store
        decoded.append(value)
    return decoded

# tournament selection


def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# crossover two parents to create two children


def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

# mutation operator


def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]

# genetic algorithm


def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best, best_eval = 0, 4  # objective(decode(bounds, n_bits, pop[0])
    # enumerate generations
    for gen in range(n_iter):
        # decode population
        decoded = [decode(bounds, n_bits, p) for p in pop]
        # evaluate all candidates in the population
        scores = []
        for element in decoded:
            if constraint(element) == True:
                scores.append(objective(element))
            else:
                scores.append(5)
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                decoded_int = [int(x) for x in decoded[i]]
                print(">%d--f(%s) = %f" %
                      (gen,  decoded_int, scores[i]))
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [best, best_eval]


# define range for input
bounds = [[0, n[0]], [0, n[1]], [0, n[2]], [0, n[3]], [0, n[4]], [0, n[5]],
          [0, n[0]], [0, n[1]], [0, n[2]], [0, n[3]], [0, n[4]], [0, n[5]],
          [0, electricity_limit0], [0, electricity_limit1]]
# define the total iterations
n_iter = 100
# bits per variable
n_bits = 50
# define the population size
n_pop = 1000
# crossover rate
r_cross = 0.8
# mutation rate
r_mut = 1.0 / (float(n_bits) * len(bounds))
# perform the genetic algorithm search
best, score = genetic_algorithm(
    objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
decoded = decode(bounds, n_bits, best)
general_usage = 0
for i in range(12):
    general_usage += decoded[i]
    decoded[i] = int(decoded[i])
decoded[12] = int(decoded[12])
decoded[13] = int(decoded[13])
print('f(%s) = %f' % (decoded, score))
print('general usage: %i' % (general_usage))
print('power generation: %i' % (decoded[12] + decoded[13]))
