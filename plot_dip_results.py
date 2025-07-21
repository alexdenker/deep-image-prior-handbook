
import os 
import yaml 

import matplotlib.pyplot as plt 
import numpy as np 

from PIL import Image 
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

load_dir = "dip_results/vanilla/random"


with open(os.path.join(load_dir, "results.yaml"), "r") as f:
    cfg = yaml.safe_load(f)

print(cfg)

markers = {
    200: 'tab:red',
    cfg["best_psnr_idx"]: 'tab:green',
    10000: 'tab:blue'
}


loss_curve = np.load(os.path.join(load_dir, "loss.npy"))
psnr_curve = np.load(os.path.join(load_dir, "psnrs.npy"))

best_reco = Image.open(os.path.join(load_dir, "best_reco.png"))
best_reco = np.asarray(best_reco)*1.0/255

ground_truth = Image.open(os.path.join(load_dir, "groundtruth.png"))
ground_truth = np.asarray(ground_truth)*1.0/255

reco_at_200 = Image.open(os.path.join(load_dir, "reco_at_200.png"))
reco_at_200 = np.asarray(reco_at_200)*1.0/255

final_reco = Image.open(os.path.join(load_dir, "final_reco.png"))
final_reco = np.asarray(final_reco)*1.0/255

fig = plt.figure(figsize=(9, 6))
gs = gridspec.GridSpec(2, 4,hspace=0.3, wspace=0.5)

# Plot loss curve
ax0 = fig.add_subplot(gs[0,0:2])
for x, color in markers.items():
    ax0.axvline(x=x, color=color, linestyle='--', linewidth=2)
ax0.loglog(loss_curve, color='black', linewidth=1)

ax0.set_title("Data Consistency", fontsize=10)
ax0.set_xlabel("Iteration", fontsize=8)
ax0.set_ylabel("Loss", fontsize=8)
ax0.tick_params(labelsize=8)

# Plot reco_at_200
ax1 = fig.add_subplot(gs[1, 0])
ax1.imshow(reco_at_200, cmap='gray', vmin=0, vmax=1)
ax1.set_title("Reconstruction after \n 200 iterations", fontsize=10)
ax1.set_xticks([])
ax1.set_yticks([])
#ax1.axis('off')
#ax1.add_patch(Rectangle((0, 0), reco_at_1000.shape[1], reco_at_1000.shape[0],
#                        linewidth=4, edgecolor=markers[1000], facecolor='none'))
for spine in ax1.spines.values():
    spine.set_edgecolor(markers[200])
    spine.set_linewidth(3)

# Plot final_reco
ax2 = fig.add_subplot(gs[1, 1])
ax2.imshow(final_reco, cmap='gray', vmin=0, vmax=1)
ax2.set_title("Final Reconstruction", fontsize=10)
#ax2.axis('off')
ax2.set_xticks([])
ax2.set_yticks([])
#ax2.add_patch(Rectangle((0, 0), final_reco.shape[1], final_reco.shape[0],
#                linewidth=4, edgecolor=markers[10000], facecolor='none'))
for spine in ax2.spines.values():
    spine.set_edgecolor(markers[10000])
    spine.set_linewidth(3)

# Plot PSNR curve
ax3 = fig.add_subplot(gs[0,2:4])
for x, color in markers.items():
    ax3.axvline(x=x, color=color, linestyle='--', linewidth=2)
ax3.semilogx(psnr_curve, color='black', linewidth=1)

ax3.set_title("PSNR Curve", fontsize=10)
ax3.set_xlabel("Iteration", fontsize=8)
ax3.set_ylabel("PSNR (dB)", fontsize=8)
ax3.tick_params(labelsize=8)

# Plot best_reco
ax4 = fig.add_subplot(gs[1, 2])
ax4.imshow(best_reco, cmap='gray', vmin=0, vmax=1)
ax4.set_title("Best Reconstruction", fontsize=10)
ax4.set_xticks([])
ax4.set_yticks([])
#ax4.axis('off')
#ax4.add_patch(Rectangle((0, 0), best_reco.shape[1], best_reco.shape[0],
#                        linewidth=4, edgecolor=markers[1500], facecolor='none'))
for spine in ax4.spines.values():
    spine.set_edgecolor(markers[cfg["best_psnr_idx"]])
    spine.set_linewidth(3)
# Plot ground truth
ax5 = fig.add_subplot(gs[1, 3])
ax5.imshow(ground_truth, cmap='gray', vmin=0, vmax=1)
ax5.set_title("Ground Truth", fontsize=10)
ax5.axis('off')

# Save figure (optional)
plt.savefig("vanilla_dip.pdf",  bbox_inches='tight')

plt.show()