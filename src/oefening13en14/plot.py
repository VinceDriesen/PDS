
import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Usage: plot.py <csv_file> <output_png>")
    sys.exit(1)

csv_file = sys.argv[1]
output_png = sys.argv[2]

df = pd.read_csv(csv_file)

plt.plot(df['intervallen'], df['tijd_main13_ns'], label='main13')
plt.plot(df['intervallen'], df['tijd_main14_ns'], label='main14')
plt.xlabel('Aantal intervallen')
plt.ylabel('Tijd (ns)')
plt.title('Vergelijking main13 vs main14')
plt.legend()
plt.grid(True)
plt.savefig(output_png)
