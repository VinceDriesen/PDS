import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) != 3:
        print("Gebruik: python3 plot.py <data.csv> <outputPath>")
        sys.exit(1)

    filename = sys.argv[1]
    output = sys.argv[2]

    data = pd.read_csv(filename)

    plt.figure(figsize=(10, 6))
    plt.plot(data["multiplier"], data["avg_ns"], marker="o", linestyle="-")
    plt.xscale("log", base=2)
    plt.xlabel("Multiplier (afstand tussen thread-offsets, bytes)")
    plt.ylabel("Gemiddelde uitvoeringstijd (ns * 10^9)")
    plt.title("False-Sharing-effect bij verschillende multipliers")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)

    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Grafiek opgeslagen als {output}")

if __name__ == "__main__":
    main()
