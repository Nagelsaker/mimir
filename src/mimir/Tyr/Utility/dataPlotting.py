import csv
from tkinter import N
from cv2 import threshold

import matplotlib
import utils
import re
import numpy as np
from mimir.Tyr.Utility.constants import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import LineCollection

state = {
    ST_STOP : "STOP",
    ST_MOVE_BACKWARD : "MOVE B",
    ST_MOVE_FORWARD : "MOVE F",
    ST_GRIP : "GRIP",
    ST_UNGRIP : "UNGRIP",
    ST_HEIGHT : "MOVE H",
    ST_TILT_DOWN : "TILT D",
    ST_TILT_UP : "TILT U",
    ST_TURN_LEFT : "TURN L",
    ST_TURN_RIGHT : "TURN R",
    ST_RL_AGENT : "RL AGENT"
}


stateToInt = {
    "STOP" : ST_STOP,
    "MOVE B" : ST_MOVE_BACKWARD,
    "MOVE F" : ST_MOVE_FORWARD,
    "GRIP" : ST_GRIP,
    "UNGRIP" : ST_UNGRIP,
    "MOVE H" : ST_HEIGHT,
    "TILT D" : ST_TILT_DOWN,
    "TILT U" : ST_TILT_UP,
    "TURN L" : ST_TURN_LEFT,
    "TURN R" : ST_TURN_RIGHT,
    "RL AGENT" : ST_RL_AGENT
}



def loadRobotPoses(pathToData):
    with open(f"{pathToData}", "r") as fp:
        reader = csv.reader(fp, delimiter=",")
        poseData = []

        for line in reader:
            if len(line) == 1 and line[0] == "None":
                poseData.append([[None, None, None], [None, None, None, None]])
                continue

            if len(line) != 9 or line[0] != "position" and line[4] != "orientation":
                raise Exception("Robot poses could not be loaded, format not recognized")

            poseData.append([list(map(float, line[1:4])), list(map(float, line[5:]))])
    return poseData

def loadLeverPoses(pathToData):
    with open(f"{pathToData}", "r") as fp:
        reader = csv.reader(fp, delimiter=",")
        poseData = []

        for line in reader:
            if len(line) == 1 and line[0] == "None":
                poseData.append([None, None, None, None])
                continue

            if len(line) != 8 or line[0] != "estimated_angle" and line[6] != "measured_position":
                raise Exception("Robot poses could not be loaded, format not recognized")

            poseData.append([float(line[1]), float(line[3]), float(line[5]), float(line[7])])
    return poseData

def loadFSMStates(pathToData):
    with open(f"{pathToData}", "r") as fp:
        reader = csv.reader(fp)
        
        stateData = []
        for line in reader:
            if line[0] == "None":
                stateData.append(None)
                continue
            
            stateData.append(state[int(line[0])])
    return stateData


def loadHandPoints(pathToData):
    '''
    HandPoints are on the format:
    idx, X, Y, Z, Visibility, Depth
    '''
    with open(f"{pathToData}", "r") as fp:
        reader = csv.reader(fp)

        pointsData = []

        for line in reader:
            if line[0] == "None":
                points.append([None, None, None, None, None])
                continue
            if len(line) % 6 != 0:
                raise Exception("HandPoints could not be loaded, format not recognized")
            points = []
            for i in range(0, len(line), 6):
                points.append(list(map(float, line[i+1:i+6])))
            pointsData.append(points)
    return pointsData


def loadTimeSteps(pathToData):
    with open(f"{pathToData}", "r") as fp:
        reader = csv.reader(fp)
        
        tStepData = []
        for line in reader:
            if line[0] == "None":
                tStepData.append(None)
                continue
            
            tStepData.append(float(line[0]))
    return tStepData

def loadLeverData(pathToData):
    with open(f"{pathToData}", "r") as fp:
        reader = csv.reader(fp)
        
        leverData = []
        for line in reader:
            if line[0] == "None":
                leverData.append(None)
                continue
            leverData.append(listStringToList(line))
    return leverData

def loadRewardSuccess(pathToData):
    with open(f"{pathToData}", "r") as fp:
        reader = csv.reader(fp)
        
        rewardData = []
        for line in reader:
            if line[0] == "None":
                rewardData.append(None)
                continue
            data = [float(line[0]),
                    float(line[1]),
                    line[2] == 'True',
                    [float(l) for l in re.sub(" +", " ", line[3].replace("[", "").replace("]", "")).split()]]
            rewardData.append(data)
    return rewardData

def listStringToList(listString):
    lst = listString[0][1:-2].split(",")
    lst = [float(l.replace("[", "").replace("]", "").replace("(", "").replace(")", "")) for l in lst]
    return lst

def cumulateRewardSuccess(rewardSuccess, timesteps):
    cumulatedRewardSeries = []
    successSeries = []
    goalSeries = []
    n = len(timesteps)
    n_rewards = len(rewardSuccess)

    prevReward = 0
    prevSuccess = 'False'
    prevGoal = 0
    cumReward = 0
    k = 0
    for i in range(n):
        if k == n_rewards:
            cumulatedRewardSeries.append(prevReward)
            successSeries.append(prevSuccess)
            goalSeries.append(prevGoal)
            continue

        time = rewardSuccess[k][0]
        if time > timesteps[i]:
            cumulatedRewardSeries.append(prevReward)
            successSeries.append(prevSuccess)
            goalSeries.append(prevGoal)
            continue
        else:
            prevReward = rewardSuccess[k][1] + cumReward
            prevSuccess = str(bool(rewardSuccess[k][2]))
            cumulatedRewardSeries.append(rewardSuccess[k][1] + cumReward)
            successSeries.append(str(bool(rewardSuccess[k][2])))
            goalSeries.append(rewardSuccess[k][3][-1])
            cumReward += rewardSuccess[k][1]
            k += 1
    
    return cumulatedRewardSeries, successSeries, goalSeries

def plotLeverAndReward(cumulatedReward, successSeries, goalSeries, leverData, timeSteps, prefix):
    T = np.array(timeSteps)
    leverData = np.array(leverData)
    cumulatedReward = np.array(cumulatedReward)
    successSeries = np.array(successSeries)

    leverAngleMeasured = np.array(leverData)[:,1]
    lever_length = LEVER_LEN
    goal_idx = np.where(successSeries == 'True')[0][0]
    goal = np.rad2deg(goalSeries[goal_idx])
    plotLeverTrajectory2D(leverAngleMeasured, lever_length, T, goal, prefix)
    
    fig, axs = plt.subplots(6,1)
    fig.set_size_inches(10, 18.5)
    fig.set_dpi(90)
    # Plot measured lever angle
    axs[0].plot(T, np.rad2deg(leverData[:,:2]))
    axs[0].legend([r"$\omega_{e}$", r"$\omega_{m}$"])
    axs[0].set(ylabel="(deg)")
    axs[0].set_title("Lever angle")
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[0].grid()

    # Plot estimated lever angle
    # axs[1].plot(T, np.rad2deg(leverData[:,0]), label=r"$\omega_{e}$")
    # axs[1].legend()
    # axs[1].set(ylabel="(deg)")
    # axs[1].set_title("Estimated lever angle")
    # axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # axs[1].grid()

    # Plot estimated lever position in x-axis
    trueLeverPosX = np.ones(len(leverData))*LEVER_POS[0]
    axs[1].plot(T, np.vstack([leverData[:,2], trueLeverPosX]).T)
    axs[1].legend([r"$l_{x,e}$", r"$l_{x,m}$"])
    axs[1].set(ylabel="X-axis (m)")
    axs[1].set_title("Estimated lever position along X-axis")
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[1].grid()

    # Plot estimated lever position in y-axis
    trueLeverPosY = np.ones(len(leverData))*LEVER_POS[1]
    axs[2].plot(T, np.vstack([leverData[:,3], trueLeverPosY]).T)
    axs[2].legend([r"$l_{y,e}$", r"$l_{y,m}$"])
    axs[2].set(ylabel="Y-axis (m)")
    axs[2].set_title("Estimated lever position along Y-axis")
    axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[2].grid()

    # Plot estimated lever position in z-axis
    trueLeverPosZ = np.ones(len(leverData))*LEVER_POS[2]
    axs[3].plot(T, np.vstack([leverData[:,4], trueLeverPosZ]).T)
    axs[3].legend([r"$l_{z,e}$", r"$l_{z,m}$"])
    axs[3].set(ylabel="Z-axis (m)")
    axs[3].set_title("Estimated lever position along Z-axis")
    axs[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[3].grid()

    # Plot cumulative reward
    axs[4].plot(T, cumulatedReward, label=r"$reward$")
    axs[4].legend()
    axs[4].set(ylabel="Reward")
    axs[4].set_title("Cumulative reward")
    axs[4].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[4].grid()

    # Plot success series
    axs[5].step(T, successSeries, label="Success")
    axs[5].legend()
    axs[5].set(ylabel="Success")
    axs[5].set(xlabel="Time (s)")
    axs[5].set_title("Goal reached?")
    axs[5].grid()

    # plt.show()
    plt.savefig(f"{prefix}_Lever_And_Reward.eps")
    

def plotTrajectory3D(manipulator, lever, T, prefix):
    n = len(T)
    X_m = manipulator[0]
    Y_m = manipulator[1]
    Z_m = manipulator[2]
    points_m = np.array(manipulator).T.reshape(-1, 1, 3)
    points_l = np.array(lever).T.reshape(-1, 1, 3)
    segments_m = np.concatenate([points_m[:-1], points_m[1:]], axis=1)
    segments_l = np.concatenate([points_l[:-1], points_l[1:]], axis=1)
    norm = plt.Normalize(T.min(), T.max())
    cmap_m=plt.get_cmap('viridis')
    cmap_l=plt.get_cmap('Greens')
    colors_m=[cmap_m(float(i)/(n-1)) for i in range(n-1)]
    colors_l=[cmap_l(float(i)/(n-1)) for i in range(n-1)]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in range(n-1):
        segii=segments_m[i]
        lii,=ax.plot(segii[:,0],segii[:,1],segii[:,2],color=colors_m[i],linewidth=2)
        lii.set_solid_capstyle('round')

    for i in range(n-1):
        segii=segments_l[i]
        lii,=ax.plot(segii[:,0],segii[:,1],segii[:,2],color=colors_l[i],linewidth=2)
        lii.set_solid_capstyle('round')

    # ax.set_box_aspect((np.ptp(points_m[0]), np.ptp(points_m[1]), np.ptp(points_m[2])))
    # ax.set_xticks(np.arange(min(X_m), max(X_m), (max(X_m)-min(X_m))/6 ))
    # ax.set_yticks(np.arange(min(Y_m), max(Y_m), (max(Y_m)-min(Y_m))/3 ))
    # ax.set_zticks(np.arange(min(Z_m), max(Z_m), (max(Z_m)-min(Z_m))/2 ))
    ax.set_xlabel("X (m)", labelpad=30)
    ax.set_ylabel("Y (m)", labelpad=10)
    ax.set_zlabel("Z (m)", labelpad=10)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.show()
    # plt.savefig(f"{prefix}_Trajectory3D.eps")


def plotTrajectory2D(psi, radius, T, prefix):
    points = np.array([np.deg2rad(psi), radius]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # norm = plt.Normalize(states.min(), states.max())
    norm = plt.Normalize(T.min(), T.max())
    lines = LineCollection(segments, cmap='viridis', norm=norm)
    lines.set_array(T)
    # lines.set_linewidth(2)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # ax.plot(np.deg2rad(psi), radius)
    fig.set_size_inches(10, 10)
    fig.set_dpi(100)
    l = ax.add_collection(lines)
    ax.set_ylabel("Radius (m)", rotation=0, labelpad=-70, size=11)
    ax.set_title("Trajectory")
    ax.set_yticks(np.arange(0, max(radius), 0.05))
    ax.set_xticks(np.arange(np.pi, -np.pi, -np.pi/4))
    ax.set_thetalim(-np.pi, np.pi)
    # ax.set_xticks(np.deg2rad([0, -45, -90, -135, 180, 135, 90, 45]))
    ax.set_rmax(max(radius)+0.01)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar = fig.colorbar(l, ax=ax, )
    cbar.ax.set_ylabel("Time (s)", rotation=-270, labelpad=20)
    # plt.show()
    plt.savefig(f"{prefix}_trajectory2D.eps")

def plotLeverTrajectory2D(psi, lever_len, T, goal_angle, prefix):
    radius = lever_len
    n = len(psi)
    points = np.array([psi, np.ones(n)*radius]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # norm = plt.Normalize(states.min(), states.max())
    norm = plt.Normalize(T.min(), T.max())
    linewidths = np.ones(len(segments))*4
    linewidths[-int(n*0.05):] = 15
    lines = LineCollection(segments, cmap='viridis', norm=norm, linewidths=linewidths)
    lines.set_array(T)
    # lines.set_linewidth(2)
    threshold = 1.4
    goal_p = np.vstack([[np.rad2deg(goal_angle+threshold), 0.],
                        [np.deg2rad(goal_angle+threshold), 13e-2],
                        [np.rad2deg(goal_angle-threshold), 0.],
                        [np.deg2rad(goal_angle-threshold), 13e-2]])
    line_goal = LineCollection([goal_p], cmap='summer', norm=norm, linestyles='dotted')

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # ax.plot(np.deg2rad(psi), radius)
    fig.set_size_inches(10, 10)
    fig.set_dpi(100)
    ax.add_collection(line_goal)
    l = ax.add_collection(lines)
    ax.set_ylabel("(m)", rotation=0, labelpad=-70, size=11)
    ax.set_title("Lever Trajectory")
    ax.set_yticks(np.arange(0, radius*3/2, 0.05))
    ax.set_xticks(np.arange(np.pi, -np.pi, -np.pi/4))
    ax.set_thetalim(-np.pi/2, np.pi/2)
    # ax.set_xticks(np.deg2rad([0, -45, -90, -135, 180, 135, 90, 45]))
    ax.set_rmax(radius+0.04)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_theta_zero_location("N")
    cbar = fig.colorbar(l, ax=ax, )
    cbar.ax.set_ylabel("Time (s)", rotation=-270, labelpad=20)
    # plt.show()
    plt.savefig(f"{prefix}_leverTrajectory2D.eps")

def plotPoseAndStates(poseData, leverPoseData, stateData, timeSteps, prefix):
    N = len(poseData)
    X = np.array([poseData[i][0][0] for i in range(N)])
    Y = np.array([poseData[i][0][1] for i in range(N)])
    Z = np.array([poseData[i][0][2] for i in range(N)])
    lever_length = LEVER_LEN
    X_l = np.array([leverPoseData[i][5] + lever_length*np.sin(-leverPoseData[i][1]) for i in range(N)])
    Y_l = np.array([0 for i in range(N)])
    Z_l = np.array([leverPoseData[i][7] + lever_length*np.cos(leverPoseData[i][1]) for i in range(N)])
    T = np.array(timeSteps)
    print(f"\nOperations lasted {T[-1]:.2f} seconds\n")

    quaternions = np.array([
        [poseData[i][1][0] for i in range(N)],
        [poseData[i][1][1] for i in range(N)],
        [poseData[i][1][2] for i in range(N)],
        [poseData[i][1][3] for i in range(N)]
    ]).T

    eulerAnglesRad = utils.euler_from_quaternion(quaternions)
    eulerAngles = np.rad2deg(eulerAnglesRad)

    radius = np.sqrt(X**2 + Y**2)
    psi = np.rad2deg(np.arctan2(Y, X))
    
    manipulator = [X, Y, Z]
    lever = [X_l, Y_l, Z_l]
    plotTrajectory3D(manipulator, lever, T, prefix)
    plotTrajectory2D(psi, radius, T, prefix)


    fig, axs = plt.subplots(5,1)
    fig.set_size_inches(10, 16.5)
    fig.set_dpi(100)

    # Plot height
    axs[0].plot(T, Z, label="Z")
    axs[0].legend()
    axs[0].set(ylabel="(m)")
    axs[0].set_title("Height")
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[0].grid()

    # Plot polar coordinates
    axs[1].plot(T, radius, label="r")
    axs[1].legend()
    axs[1].set(ylabel="Radius (m)")
    axs[1].set_title("Polar Coordinates")
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[1].grid()

    axs[2].plot(T, psi, label=r"$\beta$")
    axs[2].legend()
    axs[2].set(ylabel="Horizontal rot (Deg)")
    axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[2].grid()
    
    # Euler angles of end-effector
    axs[3].plot(T, eulerAngles)
    axs[3].legend([r"$\psi$", r"$\theta$", r"$\phi$"])
    axs[3].set(ylabel="Euler angles (Deg)")
    axs[3].set_title("Orientation")
    axs[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[3].grid()

    # Plotting State
    axs[4].step(T, stateData, label="State")
    axs[4].legend()
    axs[4].set(ylabel="States")
    axs[4].set(xlabel="Time (s)")
    axs[4].set_title("State")
    axs[4].grid()

    # Plotting State
    # axs[5].step(T, np.rad2deg(eulerAnglesRad[:-5,2] - np.arctan2(Y, X)[5:]), label="State")
    # axs[5].legend()
    # axs[5].set(ylabel="States")
    # axs[5].set(xlabel="Time (s)")
    # axs[5].set_title("State")
    # axs[5].grid()
    
    # plt.show()
    plt.savefig(f"{prefix}_Pose_and_States.eps")

if __name__ == "__main__":
    # Load data
    # prefix = "log/3_2021_12_09_"
    prefix = "log/11_2021_12_10_"
    # # pointsData = loadHandPoints(f"{prefix}HandPointsLogger.csv")
    # timeSteps = loadTimeSteps(f"{prefix}TimeStepLogger.csv")

    # # Plotting Pose:

    # prefix = "/home/simon/catkin_ws/src/mimir/src/mimir/Tyr/log/success_using_estimate/21_2022_05_23_"
    prefix = "/home/simon/catkin_ws/src/mimir/src/mimir/Tyr/log/success/9_2022_05_23_"
    poseData = loadRobotPoses(f"{prefix}RobotPoseLogger.csv")
    leverData = loadLeverData(f"{prefix}LeverPoseLogger.csv")
    stateData = loadFSMStates(f"{prefix}FSMStateLogger.csv")
    timeSteps = loadTimeSteps(f"{prefix}TimeStepLogger.csv")
    rewardSuccess = loadRewardSuccess(f"{prefix}RewardSuccessLogger.csv")
    cumulatedReward, appendedSuccess, appendedGoals = cumulateRewardSuccess(rewardSuccess, timeSteps)

    plotLeverAndReward(cumulatedReward, appendedSuccess, appendedGoals, leverData, timeSteps, prefix)
    plotPoseAndStates(poseData, leverData, stateData, timeSteps, prefix)
    print("heh")