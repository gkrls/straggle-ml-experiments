import json, os
import matplotlib.pyplot as plt
import numpy as np

data = {
  'ns' : {
    4 : {
      100 : {
        'ok' : [258490320,258490320],
        're' : [1415268,1547444],
        'to' : [0],
        'sy' : [0],
        'ol' : [0]
      },
      250 : {
        'ok' : [258490320,258490320],
        're' : [19516,20428],
        'to' : [0],
        'sy' : [0],
        'ol' : [0]
      },
    }
  },
  'sa-1500' : {
    4 : {
      250 : {
        'ok' : [246058084,246026900],
        're' : [270785728,272115792],
        'to' : [16864120,16678184],
        'sy' : [1472,1344],
        'ol' : [8192,8192]
      },
      500 : {
        'ok' : [247075964,246649224],
        're' : [111186276,112271480],
        'to' : [15219028,14122976],
        'sy' : [2368,1280],
        'ol' : [7680,7936] 
      },
      750 : {
        'ok' : [247890116,247924512],
        're' : [57539324,58863004],
        'to' : [15720084,14557828],
        'sy' : [1984,1856],
        'ol' : [7168,6912] 
      },
      1000 : {
        'ok' : [248222376,248255144,247971488],
        're' : [38144704,37000608,35801232],
        'to' : [12738788,14158328,13305364],
        'sy' : [896,896,1024],
        'ol' : [6912,6912,6912]
      }
    }
  }
}

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

PIPES = 4
sa_1500 = data['sa-1500'][PIPES]
barWidth = 0.25

# Use sorted timeouts to keep order stable
timeouts = sorted(sa_1500.keys())          # [250, 500, 750]
x = np.arange(len(timeouts))

# Bar positions: center each group at x
sa_1500_br_ok = x - barWidth
sa_1500_br_re = x
sa_1500_br_to = x + barWidth

# Values in the same order as 'timeouts'
sa_1500_ok = [np.mean(sa_1500[t]['ok']) for t in timeouts]
sa_1500_re = [np.mean(sa_1500[t]['re']) for t in timeouts]
sa_1500_to = [np.mean(sa_1500[t]['to']) for t in timeouts]

axes[0].bar(sa_1500_br_ok, sa_1500_ok, label='ok', width=barWidth, edgecolor='grey')
axes[0].bar(sa_1500_br_re, sa_1500_re, label='re', width=barWidth, edgecolor='grey')
axes[0].bar(sa_1500_br_to, sa_1500_to, label='to', width=barWidth, edgecolor='grey')

axes[0].set_title("Straggler timeout 1500us", fontweight='bold')
axes[0].set_xlabel('Retransmission timeout', fontweight='bold')
axes[0].set_ylabel('Number of packets', fontweight='bold')

# Make the x ticks the retransmission timeouts
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"{t} us" for t in timeouts])

axes[0].legend()
axes[0].grid(True, alpha=0.3)
plt.show()