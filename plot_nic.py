import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

path = sys.argv[1] if len(sys.argv) > 1 else "nic_throughput_straggle_mod_sa_rank0.csv"
path_out = Path(path).stem + ".pdf"

df = pd.read_csv(path)
df['dt'] = df['time_us'].diff()
df['tx_gbps'] = df['tx_bytes'].diff() * 8 / (df['dt'] * 1e3)
df['rx_gbps'] = df['rx_bytes'].diff() * 8 / (df['dt'] * 1e3)
df['tx_gbps'] = df['tx_gbps'].clip(upper=100)
df['rx_gbps'] = df['rx_gbps'].clip(upper=100)
# crop to where bytes actually change
changing = df['tx_bytes'].diff().fillna(0) != 0
first = changing.idxmax()
last = len(df) - 1 - changing[::-1].values.argmax()
margin = 5
df = df.iloc[max(0, first - margin):last + margin + 1].reset_index(drop=True)

# smooth
sigma = 5
df['tx_raw'] = df['tx_gbps'].copy()
df['rx_raw'] = df['rx_gbps'].copy()
df['tx_gbps'] = gaussian_filter1d(df['tx_gbps'].fillna(0), sigma=sigma)
df['rx_gbps'] = gaussian_filter1d(df['rx_gbps'].fillna(0), sigma=sigma)

# zero x-axis
t_ms = (df['time_us'] - df['time_us'].iloc[0]) / 1e3

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_ms, df['tx_raw'], alpha=0.2, color='C0', label='TX raw')
ax.plot(t_ms, df['tx_gbps'], color='C0', linewidth=2, label='TX smoothed')
ax.plot(t_ms, df['rx_raw'], alpha=0.2, color='C1', label='RX raw')
ax.plot(t_ms, df['rx_gbps'], color='C1', linewidth=2, label='RX smoothed')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Throughput (Gbps)')
ax.set_title('NIC Throughput')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(path_out, dpi=150)
plt.show()
print(f"Saved to {path_out}")