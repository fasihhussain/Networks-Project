from random import random
import networkx as nx
import matplotlib.pyplot as plt

from motif import motifCounter
from motif import motifs
import network_analysis as na
import numpy as np

file_names = ["Chesapeake", "ChesLow", "ChesMid", "ChesUp", "CrystalC", "CrystalD",
                "Maspalomas", "Michigan", "Mondego", "Narragan", "Rhode", "StMarks"]

motif_n = motifs.keys()

motif_freq = {"Chesapeake": {'S1': 179, 'S2': 152, 'S3': 0, 'S4': 908, 'S5': 210, 'D1': 12, 'D2': 3, 'D3': 21, 'D4': 37, 'D5': 7, 'D6': 0, 'D7': 6, 'D8': 0},
"ChesLow": {'S1': 196, 'S2': 178, 'S3': 5, 'S4': 816, 'S5': 175, 'D1': 15, 'D2': 9, 'D3': 26, 'D4': 77, 'D5': 9, 'D6': 0, 'D7': 7, 'D8': 5},
"ChesMid": {'S1': 276, 'S2': 273, 'S3': 16, 'S4': 857, 'S5': 171, 'D1': 20, 'D2': 9, 'D3': 26, 'D4': 73, 'D5': 11, 'D6': 0, 'D7': 4, 'D8': 2}
,"StMarks": {'S1': 699, 'S2': 519, 'S3': 0, 'S4': 1812, 'S5': 924, 'D1': 2, 'D2': 0, 'D3': 4, 'D4': 36, 'D5': 12, 'D6': 0, 'D7': 0, 'D8': 0}
,"Rhode": {'S1': 32, 'S2': 15, 'S3': 0, 'S4': 15, 'S5': 107, 'D1': 0, 'D2': 7, 'D3': 2, 'D4': 26, 'D5': 0, 'D6': 0, 'D7': 0, 'D8': 11}
,"Narragan": {'S1': 189, 'S2': 283, 'S3': 0, 'S4': 822, 'S5': 178, 'D1': 10, 'D2': 11, 'D3': 16, 'D4': 171, 'D5': 39, 'D6': 0, 'D7': 16, 'D8': 62}
,"Mondego": {'S1': 206, 'S2': 212, 'S3': 0, 'S4': 931, 'S5': 248, 'D1': 0, 'D2': 19, 'D3': 42, 'D4': 206, 'D5': 49, 'D6': 0, 'D7': 45, 'D8': 165}
,"Michigan": {'S1': 389, 'S2': 189, 'S3': 0, 'S4': 679, 'S5': 311, 'D1': 0, 'D2': 19, 'D3': 14, 'D4': 116, 'D5': 26, 'D6': 0, 'D7': 5, 'D8': 16}
,"Maspalomas": {'S1': 76, 'S2': 45, 'S3': 2, 'S4': 231, 'S5': 46, 'D1': 2, 'D2': 5, 'D3': 11, 'D4': 43, 'D5': 4, 'D6': 0, 'D7': 3, 'D8': 3}
,"CrystalD": {'S1': 25, 'S2': 65, 'S3': 0, 'S4': 327, 'S5': 102, 'D1': 13, 'D2': 2, 'D3': 1, 'D4': 72, 'D5': 10, 'D6': 0, 'D7': 9, 'D8': 12}
,"CrystalC": {'S1': 66, 'S2': 137, 'S3': 0, 'S4': 372, 'S5': 121, 'D1': 23, 'D2': 8, 'D3': 1, 'D4': 74, 'D5': 28, 'D6': 0, 'D7': 13, 'D8': 32}
,"ChesUp": {'S1': 233, 'S2': 276, 'S3': 0, 'S4': 909, 'S5': 166, 'D1': 20, 'D2': 17, 'D3': 41, 'D4': 155, 'D5': 28, 'D6': 0, 'D7': 11, 'D8': 37}
}

#"Everglades":{'S1': 1788, 'S2': 0, 'S3': 0, 'S4': 5907, 'S5': 2917, 'D1': 36, 'D2': 0, 'D3': 89, 'D4': 764, 'D5': 388, 'D6': 0, 'D7': 0, 'D8': 129}

s1 = []
s2 = []
s3 = []
s4 = []
s5 = []
d1 = []
d2 = []
d3 = []
d4 = []
d5 = []
d6 = []
d7 = [] 
d8 = []

for i in file_names:
    s1.append(motif_freq[i]['S1'])
    s2.append(motif_freq[i]['S2'])
    s3.append(motif_freq[i]['S3'])
    s4.append(motif_freq[i]['S4'])
    s5.append(motif_freq[i]['S5'])
    d1.append(motif_freq[i]['D1'])
    d2.append(motif_freq[i]['D2'])
    d3.append(motif_freq[i]['D3'])
    d4.append(motif_freq[i]['D4'])
    d5.append(motif_freq[i]['D5'])
    d6.append(motif_freq[i]['D6'])
    d7.append(motif_freq[i]['D7'])
    d8.append(motif_freq[i]['D8'])

# print(file_names, s1)
# print(map(lambda mot_list: np.array(mot_list), [s1, s2, s3, s4, s5, d1, d2, d3, d4, d5, d6, d7, d8]))

s1 = np.array(s1)
s2 = np.array(s2)
s3 = np.array(s3)
s4 = np.array(s4)
s5 = np.array(s5)
d1 = np.array(d1)
d2 = np.array(d2)
d3 = np.array(d3)
d4 = np.array(d4)
d5 = np.array(d5)
d6 = np.array(d6)
d7 = np.array(d7)
d8 = np.array(d8)

plt.bar(file_names, s1)
plt.bar(file_names, s2, bottom=s1)
plt.bar(file_names, s3, bottom=s1+s2)
plt.bar(file_names, s4, bottom=s1+s2+s3)
plt.bar(file_names, s5, bottom=s1+s2+s3+s4)
plt.bar(file_names, d1, bottom=s1+s2+s3+s4+s5)
plt.bar(file_names, d2, bottom=s1+s2+s3+s4+s5+d1)
plt.bar(file_names, d3, bottom=s1+s2+s3+s4+s5+d1+d2)
plt.bar(file_names, d4, bottom=s1+s2+s3+s4+s5+d1+d2+d3)
plt.bar(file_names, d5, bottom=s1+s2+s3+s4+s5+d1+d2+d3+d4)
plt.bar(file_names, d6, bottom=s1+s2+s3+s4+s5+d1+d2+d3+d4+d5)
plt.bar(file_names, d7, bottom=s1+s2+s3+s4+s5+d1+d2+d3+d4+d5+d6)
plt.bar(file_names, d8, bottom=s1+s2+s3+s4+s5+d1+d2+d3+d4+d5+d6+d7)

plt.legend(motif_n)
plt.xlabel("Food Web Networks")
plt.ylabel("Frequency of Each Motif")
plt.title("Motif Frequency of Each Dataset")
plt.show()
