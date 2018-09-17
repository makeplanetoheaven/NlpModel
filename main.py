# -*- coding: UTF-8 -*-

from numpy import *
from RandomSampling import *

# chineseRestaurantProcee
def chineseRestaurantProcess(n=0,alpha=0.5):
    customers = zeros(n,dtype=int)
    for i in range(n):
        print("time ------------",i+1)
        if i == 0:
            customers[i] = 1
        else:
            # statistics old tables
            tables, peopleNumPerTable = unique(customers, return_counts=True)
            if tables[0] == 0:
                tables = delete(tables, 0)
                peopleNumPerTable = delete(peopleNumPerTable, 0)
            # append a new table
            tables = append(tables, max(tables) + 1)
            peopleNumPerTable = append(peopleNumPerTable, alpha)
            # print tables and peopleNumPerTable
            print("tables:",tables)
            print("peopleNumPerTable:",peopleNumPerTable)
            # probability calculation
            probability = zeros(len(tables))
            for j in range(len(tables)):
                probability[j] = peopleNumPerTable[j]/(i+alpha)
            # print probabilityPerTable
            print("probabilityPerTable:",probability)
            # sampling
            _,tableIndex = MCMC.MetropolisHastings(p=probability,Nlmax=1)
            customers[i] = tableIndex[0]+1
            print("customer:",customers)

    return customers

if __name__ == '__main__':
    chineseRestaurantProcess(n=1000, alpha=0.5)