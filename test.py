import matplotlib.pyplot as plt
import numpy as np

labels = ['Full', 'w/o LSPM', 'w/o GSUM', 'w/o FT']
cbd_bleu = [74.63, 65.80, 61.45, 32.15]
fcb_bleu = [70.77, 61.12, 57.30, 28.40]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, cbd_bleu, width, label='CBD (BLEU)')
rects2 = ax.bar(x + width/2, fcb_bleu, width, label='FC_B (BLEU)')

ax.set_ylabel('Scores')
ax.set_title('Ablation Study on BLEU Metric')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()