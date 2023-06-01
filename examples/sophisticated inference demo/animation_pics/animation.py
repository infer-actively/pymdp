import imageio

# Figure figure when rendered in environment
# plt.savefig(f'./img/img_{self.tau}.png')

frames = []
N = 3
for t in range(8):
    image = imageio.v2.imread(f'./img_N_{N}/img_{t}.png')
    frames.append(image)

imageio.mimsave('./N_3.gif', # output gif
            frames,          # array of input frames
            fps = 2)         # optional: frames per second
