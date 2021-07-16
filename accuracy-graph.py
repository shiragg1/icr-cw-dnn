#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

#set up path to directory, replace with path on your computer
p = Path('/home/shira/Documents/icr/icr-cw-dnn/')

labels = ['0.5', '1', '2', '3', 'Random (0.5 - 3)']
raw_data = [53.61, 53.04, 50.37, 53.03, 48.13]
fft_data = [98.09, 94.43, 61.76, 49.96, 62.93]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, raw_data, width, label='Raw Data')
rects2 = ax.bar(x + width/2, fft_data, width, label='Spectrum of Data')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percent Accuracy')
ax.set_xlabel('Noise')
ax.set_title('Performance of DNNs with Varying Noise Levels')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.savefig(p / "dnn-accuracy")