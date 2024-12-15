import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img1 = mpimg.imread('results/s14/PCA_all.png')
img2 = mpimg.imread('results/s14/tsne_all.png')
img3 = mpimg.imread('results/s14/umap_all.png')

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(img1)
axs[0].set_title('PCA')
axs[0].axis('off')

axs[1].imshow(img2)
axs[1].set_title('t-SNE')
axs[1].axis('off')

axs[2].imshow(img3)
axs[2].set_title('UMAP')
axs[2].axis('off')

plt.tight_layout()

plt.savefig('results/s14/s14_all.png', dpi=300)

plt.show()