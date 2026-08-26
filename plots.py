import os
import matplotlib.pyplot as plt
import numpy as np


def plot_abundance(ground_truth, estimated, em, save_dir,  rmse_cls):
    plt.figure(figsize=(12, 6), dpi=300)
    for i in range(em):
        plt.subplot(2, em, i + 1)
        plt.imshow(ground_truth[:, :, i], cmap='jet')
        plt.title(f"GT Class {i + 1}")
        plt.axis('off') 
    for i in range(em):
        plt.subplot(2, em, em + i + 1)
        plt.imshow(estimated[:, :, i], cmap='jet')
        plt.title(f"Pred Class {i + 1} | RMSE={rmse_cls[i]:.4f}")
        plt.axis('off')   
    plt.tight_layout()
    plt.savefig(save_dir + "abundance.png")

def plot_endmembers(target, pred, em, save_dir, sad_cls):
    fig, axes = plt.subplots(1, em, figsize=(5 * em, 4), dpi=300)
    if em == 1:
        axes = [axes]
    for i in range(em):
        target_norm = target[:, i] / np.max(target[:, i])
        pred_norm = pred[:, i] / np.max(pred[:, i])
        ax = axes[i]
        ax.plot(pred_norm, color='b', linewidth=1.0, label="Extracted")
        ax.plot(target_norm, color='red', linewidth=1.0, label="GT")
        ax.set_ylim([0.0, 1.0])
        ax.set_xticks([])
        ax.set_title(f"Class {i + 1} | SAD={sad_cls[i]:.4f}")
        if i == 0:
            ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "endmembers.png"), bbox_inches="tight")
    plt.close(fig)

def plot_intra_weights(weights_list, save_dir):
    print("正在绘制类内权重热力图...")
    save_path = os.path.join(save_dir, "intra_weights_maps")
    os.makedirs(save_path, exist_ok=True)

    for p_idx, weight_map in enumerate(weights_list):
        K = weight_map.shape[0]
        cols = 4 if K > 4 else K
        rows = (K + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150)
        fig.suptitle(f"Endmember {p_idx + 1} Intra-Bundle Weights (Total {K} signatures)", fontsize=16)
        if K == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for k in range(K):
            ax = axes[k]
            w_img = weight_map[k, :, :]
            im = ax.imshow(w_img, cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f"Signature {k + 1}", fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for k in range(K, len(axes)):
            axes[k].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        filename = f"Endmember_{p_idx + 1}_weights.png"
        plt.savefig(os.path.join(save_path, filename))
        plt.close()
        print(f"已保存端元 {p_idx + 1} 的权重图: {filename}")
