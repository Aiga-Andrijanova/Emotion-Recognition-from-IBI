from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
import os

MAX_SEQ_LEN = 15
N = 9750  # ???

MEMMAP_FILE = np.memmap('AMIGOS', dtype='float16', mode='w+', shape=(N, MAX_SEQ_LEN))
MEMMAP_FILE.flush()

PATH = "D:/Universitātes darbi/Bakalaura darbs/Datasets/AMIGOS/Data_Preprocessed/"
FREQ = 128  # Hz

memmap_i = 0

for filename in os.listdir(PATH):

    DATA = loadmat(PATH + filename)

    # There are 20 videos, 0-15 short videos, 16-19 long
    for video in range(0, 15):

        arousal = DATA['labels_selfassessment'][0][video][0, 0]
        valence = DATA['labels_selfassessment'][0][video][0, 1]

        LeadII = DATA['joined_data'][0][video][:, 14]
        LeadIII = DATA['joined_data'][0][video][:, 15]

        # DATA['joined_data'][0][Video NR. (0-19)][:,14] ECG right arm lead (Lead II) in mV
        # DATA['joined_data'][0][Video NR. (0-19)][:,14] ECG left arm lead (Lead III) in mV

        M_train = []
        F_train = []
        R_train = []
        MFR_train = []

        IBI_list = []  # Or RR interval
        QRS_list = []

        ComplexLead = np.zeros(len(LeadII)-2)
        for i in range(0, len(LeadII)-2):
            ComplexLead[i] = 0.5 * (np.abs(LeadII[i + 2] - LeadII[i]) + np.abs(LeadIII[i + 2] - LeadIII[i])) / 100
            if ComplexLead[i] < 0.8:  # "Hack"
                ComplexLead[i] = 0

        # plt.plot(LeadII)
        # plt.plot(LeadIII)
        # plt.plot(ComplexLead)
        # plt.show()

        M = 0.6 * np.max(ComplexLead[:5*FREQ])  # Max val in the first 5 seconds
        M_max = M
        MM = [M, M, M, M, M]

        Dist = 0
        for i in range(0, int(0.35 * FREQ)):
            Dist = Dist + ComplexLead[i+1] - ComplexLead[i]
        F = Dist / 0.35  # mean pseudo-spatial velocity for the first 350 ms

        RR_EXPECTED = 0.65  # Normal RR interval = 0.6-1.2 seconds
        R = 0
        RR = np.zeros(5)

        MFR = M  # + R + F

        M_train.append(M)
        F_train.append(F)
        R_train.append(R)
        MFR_train.append(MFR)

        DetectionTime = -0.2

        for i in range(0, len(ComplexLead)-1):
            Time = i / FREQ
            if ComplexLead[i] >= MFR and Time >= DetectionTime + 0.2:  # QRS complex detected, MM and RR buffer renewal proceeds
                if DetectionTime == -0.2:
                    DetectionTime = Time

                QRS_list.append(Time)
                if len(QRS_list) >= 2:
                    RR[0] = RR[1]
                    RR[1] = RR[2]
                    RR[2] = RR[3]
                    RR[3] = RR[4]
                    RR[4] = QRS_list[-1] - QRS_list[-2]
                    IBI_list.append(QRS_list[-1] - QRS_list[-2])

                MM[0] = MM[1]
                MM[1] = MM[2]
                MM[2] = MM[3]
                MM[3] = MM[4]

                new_M4 = 0.6 * np.max(ComplexLead[i])

                if new_M4 > 1.5 * MM[4]:
                    MM[4] = 1.1 * MM[4]
                else:
                    MM[4] = new_M4
                DetectionTime = Time

                M = np.average(MM)
                M_max = M
                M_train.append(M)
            elif DetectionTime + 0.2 <= Time <= DetectionTime + 1.2 and len(QRS_list) >= 1:  # decreasing M value
                M = M_max * (1 - (Time - DetectionTime - 0.2) * 0.4)
                M_train.append(M)
            else:
                M_train.append(M)

            if len(QRS_list) >= 1:
                if DetectionTime <= Time <= DetectionTime + 2/3 * RR_EXPECTED:
                    R = 0
                    R_train.append(R)
                elif DetectionTime + 2/3 * np.mean(RR) <= Time <= DetectionTime + np.mean(RR):
                    R = 0 - ((Time - DetectionTime - (2/3 * np.mean(RR))) * 0.3)
                    R_train.append(R)
                else:
                    R_train.append(R)
            else:
                R_train.append(R)

            if i > int(FREQ * 0.35):
                # F = F + (max(Y in latest 50 ms in the 350 ms interval) -
                # max(Y in earliest 50 ms in the 350 ms interval))/150
                F = F + (np.max(ComplexLead[i - int(0.05 * FREQ): i]) -
                         np.max(ComplexLead[i - int(0.35 * FREQ): i - int(0.30 * FREQ)])) / 150
                F_train.append(F)
            else:
                F_train.append(F)

            MFR = M  # + F + R
            MFR_train.append(MFR)

        # time = np.arange(0, len(MFR_train)/FREQ, 1/FREQ)
        # plt.plot(time, M_train, label='M')
        # plt.plot(time, F_train, label='F')
        # plt.plot(time, R_train, label='R')
        # plt.plot(time, MFR_train, label='MFR')
        # plt.plot(time, ComplexLead, label='Complex Lead')
        # plt.legend(loc='upper left', borderaxespad=0.)
        # plt.show()

        IBI_train = []
        IBI_sum = 0
        for IBI in IBI_list:
            IBI_train.append(IBI)

            IBI_sum = IBI_sum + IBI
            if IBI_sum >= 5:
                IBI_sum = 0

                MEMMAP_FILE = np.memmap('AMIGOS', dtype='float16', mode='r+', offset=2*(memmap_i * MAX_SEQ_LEN))

                j = 0
                for val in IBI_train:
                    MEMMAP_FILE[j] = val
                    j = j + 1

                MEMMAP_FILE.flush()

                memmap_i = memmap_i + 1
                IBI_train.clear()