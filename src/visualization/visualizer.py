import sqlite3
import math
import matplotlib.pyplot as plt
from .radar_chart import radar_factory


class Visualizer:
    def __init__(self, analysis_database_path):
        self.analysis_database_path = analysis_database_path

    def visualize_density_accuracy(self):
        sequences = {}
        algorithms = []
        con = sqlite3.connect(self.analysis_database_path)
        cur = con.cursor()
        cur.execute('SELECT sequence, detector_type, feature_type, points, error FROM reconstruction WHERE detector_type in ("SIFT","SURF","FAST") and feature_type in ("SIFT","SURF","DAISY");')
        rows = cur.fetchall()
        for row in rows:
            if row[0] not in sequences:
                sequences[row[0]] = {}
            algorithm = f'{row[1]} - {row[2]}'
            if algorithm not in algorithms:
                algorithms.append(algorithm)
            sequences[row[0]][algorithm] = (row[3], row[4])
        algorithms = sorted(algorithms)
        data = [
            ('Point Cloud Size (Normalized)',
             [[sequences[s].get(a, (0, 0))[0]/max([sequences[s].get(a1, (0, 0))[0] for a1 in algorithms]) for s in sequences] for a in algorithms]),
            ('Reprojection Error',
             [[sequences[s].get(a, (0, 0))[1] for s in sequences] for a in algorithms]),
        ]
        con.close()
        N = len(sequences)
        theta = radar_factory(N, frame='polygon')
        spoke_labels = list(sequences.keys())
        fig, axs = plt.subplots(figsize=(20, 10), nrows=1, ncols=2,
                                subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=0.2, hspace=0.5, top=0.85, bottom=0.05)
        cm = plt.get_cmap('tab20')
        colors = [cm(i) for i in range(len(algorithms))]
        markers = ['o', '^', 'v', '<', '>', 's',
                   '+', 'x', 'd', '1', '2', '3',
                   '4', 'h', 'p', '|', '_', 'D', 'H']
        for ax, (title, case_data) in zip(axs.flat, data):
            ax.set_title(title, weight='bold', size='large',
                         horizontalalignment='center', verticalalignment='center')
            for d, color, marker in zip(case_data, colors, markers):
                ax.plot(theta, d, color=color, marker=marker, linestyle='-')
            ax.set_varlabels(spoke_labels)
        labels = algorithms
        legend = fig.legend(labels, loc='upper center',
                            fontsize='medium', ncol=min(len(algorithms), 3))
        return fig

    def visualize_trade_off(self):
        sequences = {}
        algorithms = []
        con = sqlite3.connect(self.analysis_database_path)
        cur = con.cursor()
        cur.execute('SELECT DISTINCT sequence, detector_type1, feature_type1, points1, error1 FROM ranking WHERE detector_type1 in ("SIFT","SURF","FAST") and feature_type1 in ("SIFT","SURF","DAISY") ORDER BY sequence, detector_type1, feature_type1, points1;')
        rows = cur.fetchall()
        for row in rows:
            if row[0] not in sequences:
                sequences[row[0]] = {}
            algorithm = f'{row[1]} - {row[2]}'
            if algorithm not in algorithms:
                algorithms.append(algorithm)
            if algorithm not in sequences[row[0]]:
                sequences[row[0]][algorithm] = ([0], [0])
            sequences[row[0]][algorithm][0].append(row[3])
            sequences[row[0]][algorithm][1].append(row[4])
        algorithms = sorted(algorithms)
        data = [(s, [(sequences[s].get(a, ([0], [0]))[0], sequences[s].get(a, ([0], [0]))[1])
                for a in algorithms])
                for s in sequences
                ]
        con.close()
        N = len(sequences)
        fig, axs = plt.subplots(
            figsize=(20, 15), nrows=math.ceil(N / 2), ncols=2)
        fig.subplots_adjust(wspace=0.2, hspace=0.5, top=0.85, bottom=0.05)
        cm = plt.get_cmap('tab20')
        colors = [cm(i) for i in range(len(algorithms))]
        markers = ['o', '^', 'v', '<', '>', 's',
                   '+', 'x', 'd', '1', '2', '3',
                   '4', 'h', 'p', '|', '_', 'D', 'H']
        for ax, (title, case_data) in zip(axs.flat, data):
            ax.set_title(title, weight='bold', size='large',
                         horizontalalignment='center', verticalalignment='center')
            for d, color, marker in zip(case_data, colors, markers):
                ax.plot(d[0], d[1], color=color, marker=marker, linestyle='-')
                ax.set_xlabel('Point Cloud Size')
                ax.set_ylabel('Reprojection Error')
        labels = algorithms
        legend = fig.legend(labels, loc='upper center',
                            fontsize='medium', ncol=min(len(algorithms), 3))
        return fig
