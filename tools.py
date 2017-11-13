#  ce fichier permet les colonnes qui sont "catégoriques" ou "binaires"

names_str = "id,target,ps_ind_01,ps_ind_02_cat,ps_ind_03,ps_ind_04_cat,ps_ind_05_cat,ps_ind_06_bin,ps_ind_07_bin,ps_ind_08_bin,ps_ind_09_bin,ps_ind_10_bin,ps_ind_11_bin,ps_ind_12_bin,ps_ind_13_bin,ps_ind_14,ps_ind_15,ps_ind_16_bin,ps_ind_17_bin,ps_ind_18_bin,ps_reg_01,ps_reg_02,ps_reg_03,ps_car_01_cat,ps_car_02_cat,ps_car_03_cat,ps_car_04_cat,ps_car_05_cat,ps_car_06_cat,ps_car_07_cat,ps_car_08_cat,ps_car_09_cat,ps_car_10_cat,ps_car_11_cat,ps_car_11,ps_car_12,ps_car_13,ps_car_14,ps_car_15,ps_calc_01,ps_calc_02,ps_calc_03,ps_calc_04,ps_calc_05,ps_calc_06,ps_calc_07,ps_calc_08,ps_calc_09,ps_calc_10,ps_calc_11,ps_calc_12,ps_calc_13,ps_calc_14,ps_calc_15_bin,ps_calc_16_bin,ps_calc_17_bin,ps_calc_18_bin,ps_calc_19_bin,ps_calc_20_bin"

names = names_str.split(",")

# indexes = []
#
# for i in range(len(names)):
#     name = names[i]
#     print(name[len(name)-3:len(name)])
#     if name[len(name)-3:len(name)] == "cat":
#         indexes.append(i-2)
# print(indexes)


# indexes = []
#
# for i in range(len(names)):
#     name = names[i]
#     print(name[len(name)-3:len(name)])
#     if name[len(name)-3:len(name)] not in ["cat", "bin"]:
#         indexes.append(i)
# print(indexes)

import matplotlib.pyplot as plt
import numpy as np
y_test = np.loadtxt("y_test")
y_pred = np.loadtxt("y_pred")
y_train = np.loadtxt("y_train")
y_pred_train = np.loadtxt("y_pred_train")


def to_csv(y_pred, ids):
    import csv
    with open('my_answer.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['id', 'target'])
        for i in range(len(y_pred)):
            spamwriter.writerow([ids[i], y_pred[i]])


def plot_accuracy(a, p):
    accuracies = []
    for i in range(1, 100):

        pred = p > float(i)/100

        accuracies.append(np.mean(pred == (a> 0.5)))
    plt.plot(range(1, 100), accuracies)
    print(accuracies)
    plt.show()


def plot_cap_curve(a, p):

    number_of_pos = int(np.sum(a))
    index = np.argsort(p)[::-1]

    capteurs = [(a[index[i]] > 0.5) for i in range(len(index))]
    capteurs = 100*np.cumsum(capteurs)/len(p)

    crystal_ball = np.zeros(len(p))
    crystal_ball[:number_of_pos] = np.ones(number_of_pos)
    crystal_ball = 100*np.cumsum(crystal_ball)/len(p)

    random = np.ones(len(p))
    random = 100*np.cumsum(random)*number_of_pos/(len(p)**2)

    t = np.linspace(0, 100, len(p))
    plt.plot(t, capteurs)
    plt.plot(t, random)
    plt.plot(t, crystal_ball)

    plt.show()

plot_cap_curve(y_test, y_pred)
plot_cap_curve(y_train, y_pred_train)
