import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


def make_movie(filename, mov, caxis, dpi=100):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig = plt.figure()
    plt.imshow(mov[0], interpolation='nearest')
    plt.clim(caxis)

    with writer.saving(fig, filename, dpi):
        for i in range(1,mov.shape[0]):
            plt.imshow(mov[i], interpolation='nearest')
            plt.clim(caxis)
            writer.grab_frame()