import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def read_csv_data(csv_file):
    filenames = []
    translations = []
    rotations = []
    euler_angles = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader) 

        for row in reader:
            try:
                filenames.append(row[-1])  
                translations.append([float(row[0]), float(row[1]), float(row[2])])
                rotations.append([float(row[3]), float(row[4]), float(row[5])])
                euler_angles.append([float(row[6]), float(row[7]), float(row[8])])
            except (IndexError, ValueError) as e:
                print(f"Error processing row: {row}, error: {e}")
                continue  

    return filenames, translations, rotations, euler_angles


def plot_3d_data(filenames, translations, rotations, euler_angles):
    fig = plt.figure(figsize=(18, 6))

    
    ax1 = fig.add_subplot(131, projection='3d')
    for i in range(len(filenames)):
        ax1.scatter(translations[i][0], translations[i][1], translations[i][2], marker='o', label=f"{filenames[i]}")
    ax1.set_title('Translation Vectors')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    
    ax2 = fig.add_subplot(132, projection='3d')
    for i in range(len(filenames)):
        ax2.scatter(rotations[i][0], rotations[i][1], rotations[i][2], marker='^', label=f"{filenames[i]}")
    ax2.set_title('Rotation Vectors')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()

  
    ax3 = fig.add_subplot(133, projection='3d')
    for i in range(len(filenames)):
        ax3.scatter(euler_angles[i][0], euler_angles[i][1], euler_angles[i][2], marker='s', label=f"{filenames[i]}")
    ax3.set_title('Euler Angles')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_file = '/home/harshit/Desktop/project_data/pose/plot_points3/build/output_data.csv'  # Replace with your CSV file path
    filenames, translations, rotations, euler_angles = read_csv_data(csv_file)
    plot_3d_data(filenames, translations, rotations, euler_angles)

