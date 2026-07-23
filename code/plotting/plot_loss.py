import os
import matplotlib.pyplot as plt

# Plotting Loss


def plot_loss(file_name, units):
    train_loss = []
    test_loss = []

    with open(f"./data/plot-data/{file_name}.txt", "r") as loss:
        MSE_Data = loss.read().splitlines()

        for i in MSE_Data:
            train_loss.append(float(i.split(',')[0]))
            test_loss.append(float(i.split(',')[1]))

    plt.plot(train_loss, color='blue', linestyle='-', label='train')
    plt.plot(test_loss, color='red', linestyle='--', label='test')
    plt.legend()
    plt.title(f'Absorption MAE ({units}) vs Epoch ({len(train_loss)})')
    plt.xlabel('Epoch')
    plt.ylabel(f'Absorption MAE ({units})')

    plt.savefig(f'./plots-visuals/MAE-plot-{units}.png')
    plt.close()


def plot_error_histogram(outliers, dataset):
    buckets = list(range(len(outliers)))

    plt.figure(figsize=(9, 5))

    # 1. width=0.6 creates nice, obvious spacing between the bars
    bars = plt.bar(buckets, outliers, width=0.6,
                   edgecolor="black", color="#3498db")

    # 2. Log scale so the '1' count and '1600' count both fit on screen
    plt.yscale('log')

    # 3. Remove x-axis tick labels entirely (keeps empty ticks or cleans the axis)
    plt.xticks(buckets, labels=[])

    # 4. Keep exact value labels on top of the bars
    for bar, val in zip(bars, outliers):
        if val > 0:
            # Places the count slightly above the bar top
            plt.text(bar.get_x() + bar.get_width() / 2, val * 1.2,
                     str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xlabel("MAE Buckets")
    plt.ylabel("# Molecules (Log Scale)")
    plt.title(
        f"Histogram from MAE of Model Predictions vs Actual Wavelength ({dataset})")

    # Give enough Y padding so the '1600' text label doesn't get cut off at the very top
    plt.ylim(bottom=0.5, top=max(outliers) * 5)

    plt.tight_layout()
    plt.savefig(f'./plots-visuals/histogram-plot-{dataset}.png')
    plt.close()
