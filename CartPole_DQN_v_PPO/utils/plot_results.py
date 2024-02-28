import numpy as np
import matplotlib.pyplot as plt

def plot_results(values, title = '', run_experiments = False, num_experiments = 3):
    if not run_experiments:
        results = values

        fig, ax = plt.subplots(nrows=1, ncols = 2, figsize = (12,5))
        fig.suptitle(title)

        # Plot of reward
        ax[0].plot(results, label = 'Score per run')
        ax[0].axhline(195, c='red', ls='--', label = 'goal')
        ax[0].set_xlabel('Episodes')
        ax[0].set_ylabel('Reward')
        x = range(len(results))
        ax[0].legend()

        try:
            z = np.polyfit(x, results, 1)
            p = np.poly1d(z)
            ax[0].plot(x,p(x),"--", label='trend')
        except:
            print('')
        
        # Histogram of results
        ax[1].hist(results[-100:])
        ax[1].axvline(195, c='red', label='goal')
        ax[1].set_xlabel('Scores per Last 100 Episodes')
        ax[1].set_ylabel('Frequency')
        ax[1].legend()
        plt.show()

    else:
        experimental_results = values

        fig, ax = plt.subplots(3, 2, figsize = (12,12))
        fig.suptitle(title)

        for i in range(num_experiments):
            for j in range(2):
                if j == 0:
                    ax[i,j].plot(experimental_results[i], label = 'Score per run')
                    ax[i,j].axhline(195, c='red', ls='--', label = 'goal')
                    ax[i,j].set_xlabel('Episodes')
                    ax[i,j].set_ylabel('Reward')
                    x = range(len(experimental_results[i]))
                    ax[i,j].legend()
                    

                    try:
                        z = np.polyfit(x, experimental_results[i], 1)
                        p = np.poly1d(z)
                        ax[i,j].plot(x,p(x),"--", label='trend')
                    except:
                        print('')

                else:
                    ax[i,j].hist(experimental_results[i][-100:])
                    ax[i,j].axvline(195, c='red', label='goal')
                    ax[i,j].set_xlabel('Scores per Last 100 Episodes')
                    ax[i,j].set_ylabel('Frequency')
                    ax[i,j].legend()
        
        return fig
        
        
        