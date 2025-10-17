import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) != 3:
        print("Gebruik: python3 main.py <data.csv> <outputPath>")
        sys.exit(1)

    filename = sys.argv[1]
    output = sys.argv[2]
    data = pd.read_csv(filename)

    plt.figure(figsize=(10,6))
    plt.plot(data["size_MB"], data["avg_ns"], marker="o", linestyle="-")
    plt.xscale("log")
    plt.xlabel("Geheugengrootte (MB)")
    plt.ylabel("Gemiddelde tijd per element (ns)")
    plt.title("Cache performance")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)

    plt.savefig(output, dpi=300, bbox_inches="tight")
    print("Grafiek opgeslagen als plot.png")

if __name__ == "__main__":
    main()
