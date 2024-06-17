import numpy as np
import matplotlib.pyplot as plt


class LearningCurve:
    def __init__(self, dists, epoch):
        self.dists = dists
        self.seeds = [0, 1, 3]
        self.epoch = epoch
        self.curve_dict = {}
        for dist in dists:
            self.load_data(dist)


    def load_data(self, dist):
        curve = []
        for seed in self.seeds:
            path = "train_log_dist/%s_id%d_%d(%d)_%s(%s)_cam%d(%d)" % ('sim', 0, 8000, seed, 'PE', 'arm', dist * 1000, seed)
            train_log = np.loadtxt(path + "/log_train.txt")[-self.epoch:] # shape: (epoch,)
            val_log = np.loadtxt(path + "/log_val.txt")[-self.epoch:] # shape: (epoch,)
            log = np.vstack((train_log, val_log)) # shape: (2, epoch)

            curve.append(log)

        curve = np.array(curve) # shape: (3, 2, epoch)
        print(curve.shape)
        self.curve_dict[dist] = curve # dictionary: {dist: (3, 2, epoch)}

            

    def plot_curves(self):
        plt.figure(figsize=(8, 5))
        plt.style.use('seaborn-v0_8-whitegrid')
        x = np.arange(self.epoch)
        for dist in self.dists:
            y = np.mean(self.curve_dict[dist][:, 1], axis=0) * (dist * dist) # shape: (epoch,)
            yerr = np.std(self.curve_dict[dist][:, 1], axis=0) * (dist * dist) # shape: (epoch,)

            # plt.plot(x, np.log(y),label="cam_dist=%d mm" % (dist * 1000))
            # plt.fill_between(x, np.log(y - yerr), np.log(y + yerr), alpha=0.8)

            plt.plot(x, y, label="Camera Distance : %d mm" % (dist * 1000))
            plt.fill_between(x, y - yerr, y + yerr, alpha=0.5)

        # plt.yscale("log")

        plt.xlabel("epoch * 1000")
        plt.ylabel("loss")

        custom_ticks = np.arange(0.01, 0.004, 5)
        # plt.yticks(custom_ticks, [str(tick) for tick in custom_ticks])
        plt.legend()
        plt.title("Validation Loss (Normalized)")
        plt.tight_layout()
        plt.savefig("train_log_dist/report/learning_curve.png")
        plt.show()

    def plot_images(self):
        seed = self.seeds[0]
        # short_dists = [0.8, 1.2]
        # cols = short_dists
        cols = self.dists
        rows = ['0', '1000', '10000', '100000', 'best', 'gt']
        rows_label = ['Epoch: 0', 'Epoch: 1k', 'Epoch: 10k', 'Epoch: 100k', 'Best Result', 'Ground Truth']
        cols_label = ['Distance = 800 mm', 'Distance = 1000 mm', 'Distance = 1200 mm']
        fig, axs = plt.subplots(len(rows), len(cols), sharex=True, sharey=True, figsize=(12, 6))
        fig.subplots_adjust(hspace=0.)
        plt.style.use('seaborn-v0_8-whitegrid')

        for i, dist in enumerate(cols):
            for j, row in enumerate(rows):

                path = "train_log_dist/%s_id%d_%d(%d)_%s(%s)_cam%d(%d)" % ('sim', 0, 8000, seed, 'PE', 'arm', dist * 1000, seed) + "/image/%s.png" % row
                img = plt.imread(path)[:, :400, :]
                # print(img.shape)
                axs[j, i].imshow(img)
                # axs[j, i].axis('off')
                axs[j, i].set_xticks([])
                axs[j, i].set_yticks([])

        for ax, col in zip(axs[0], cols_label):
            ax.set_title(col, size='large')
        for ax, row in zip(axs[:,0], rows_label):
            ax.set_ylabel(row, rotation=0, labelpad=50) #  size='small'
        fig.tight_layout()
        plt.savefig("train_log_dist/report/training.png")
        plt.show()

    def plot_results(self):
        dist_best_loss = []
        dist_best_tloss = []
        for dist in self.dists:
            best_loss_long_seeds = np.min(self.curve_dict[dist][:, 1], axis=1) * dist * dist # shape: (3,)
            best_loss_mean = np.mean(best_loss_long_seeds)
            best_loss_std = np.std(best_loss_long_seeds)
            dist_best_loss.append((best_loss_mean, best_loss_std))

            best_tloss_long_seeds = np.min(self.curve_dict[dist][:, 0], axis=1) * dist * dist # shape: (3,)
            best_tloss_mean = np.mean(best_tloss_long_seeds)
            best_tloss_std = np.std(best_tloss_long_seeds)
            dist_best_tloss.append((best_tloss_mean, best_tloss_std))

        # plt.figure(figsize=(12, 6))
        plt.style.use('seaborn-v0_8-whitegrid')
        # plt.style.use('seaborn-whitegrid')
        width = 0.3    # the width of the bars: can also be len(x) sequence
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(self.dists))
        y = [loss for loss, _ in dist_best_loss]
        yerr = [std for _, std in dist_best_loss]
        yt = [loss for loss, _ in dist_best_tloss]
        yterr = [std for _, std in dist_best_tloss]

        np.savetxt("train_log_dist/report/best_loss.txt", np.array([y, yerr, yt, yterr]))

        # bar chart with error bars
        rects1 = ax.bar(x+width/2, y, yerr=yerr, capsize=5, label='Validation Loss', width=width)
        rects2 = ax.bar(x-width/2, yt, yerr=yterr, capsize=5, label='Training Loss', width=width)

        # plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5)
        plt.xticks(x, [str(dist * 1000) for dist in self.dists])
        plt.xlabel("Camera Distance (mm)")
        plt.ylabel("Normalized MSE Loss")
        plt.title("Minimal Training Loss and Validation Loss (Normalized)")
        
        plt.legend()
        plt.savefig("train_log_dist/report/best_loss.png")
        plt.show()


if __name__ == "__main__":
    dists = [0.8, 1.0, 1.2]
    lc = LearningCurve(dists, 398)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    lc.plot_curves()
    lc.plot_images()
    lc.plot_results()

