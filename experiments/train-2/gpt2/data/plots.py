import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

script_dir = Path(__file__).parent

data = {
  "time" : {},
  "packets" : {
    "sa" : {
      "natural" : {
        25: {"res_n":29620679,"res_k":3256481426,"ok":2737060586,"ok_to":10807891,"re":6437,"re_to":287248245,"re_out":3237954,"old_in":38641711,"old_out":29894074+6589020,"syn_in":5410083,"syn_out":4634305},
      },
      "straggle": {
         5: {"res_n":0,      "res_k":3535756854,"ok":2929216156,"ok_to":17272103,"re":5325,     "re_to":430582943,"re_out":2147772, "old_in":4866518,"old_out":4581392+257143,"syn_in":15486,"syn_out":14362},
        25: {"res_n":0,      "res_k":3535784352,"ok":2940183144,"ok_to":6304425,"re":5273,      "re_to":517810229,"re_out":6658521, "old_in":5985792,"old_out":5772662+213130,"syn_in":1920,"syn_out":1641},
        50: {"res_n":0,      "res_k":3535783756,"ok":2940566423,"ok_to":5928440,"re":201256633, "re_to":206530747,"re_out":1746410, "old_in":6010247,"old_out":5767938+241594,"syn_in":2761,"syn_out":2241},
        75: {"res_n":60,     "res_k":3535770868,"ok":2946064098,"ok_to":425746, "re":655953794, "re_to":30944135, "re_out":9711786, "old_in":6104957,"old_out":5830463+261189,"syn_in":10745,"syn_out":9607},
        100:{"res_n":2918862,"res_k":3532865490,"ok":2946668233,"ok_to":305762, "re":1109214743,"re_to":22413261, "re_out":9195524, "old_in":5985412,"old_out":5681476+303936,"syn_in":1404,"syn_out":1404},
        125:{"res_n":2918862,"res_k":3532865490,"ok":2946968060,"ok_to":5935,   "re":1532690720,"re_to":23764888, "re_out":15993296,"old_in":5985408,"old_out":392623+5592785,"syn_in":1472,"syn_out":1472}
      }
    },
    "su" : {
      "natural" : {"res_n":3535784352,"res_k":0,"ok":3520818668,"ok_to":14965684,"re":7728,"re_to":455257781, "re_out":357053,"old_in":0,"old_out":0,"syn_in":0,"syn_out":0},
      "straggle": {"res_n":3535784352,"res_k":0,"ok":3528580494,"ok_to":7203858, "re":2455,"re_to":7625910118,"re_out":302504,"old_in":0,"old_out":0,"syn_in":0,"syn_out":0},
    },
  }
}
stos = [5,25,50,75,100,125]

for sto in stos:
  with open(script_dir / f"gpt2_ga5_1epoch_sa_{sto}-1_straggl_3.15.50-200.json") as f:
    data["time"].setdefault("sa", {}).setdefault("straggle", {})[sto] = json.load(f)
with open(script_dir / "gpt2_ga5_1epoch_sa_25-1_natural.json") as f:
  data["time"].setdefault("sa", {}).setdefault("natural", {})[25] = json.load(f)
with open(script_dir / ".." / "su" / "gpt2_ga5_1epoch_su_natural.json") as f:
  data["time"].setdefault("su", {})["natural"] = json.load(f)
with open(script_dir / ".." / "su" / "gpt2_ga5_1epoch_su_straggl_3.15.50-200.json") as f:
  data["time"].setdefault("su", {})["straggle"] = json.load(f)


fig, axs = plt.subplots(2, 3, figsize=(18,10))

# epoch time
sa_times = [data["time"]["sa"]["straggle"][sto]["epochs"]["0"]["epoch_time"] for sto in stos]
su_natural_time  = data["time"]["su"]["natural"]["epochs"]["0"]["epoch_time"]
su_straggle_time = data["time"]["su"]["straggle"]["epochs"]["0"]["epoch_time"]
axs[0,0].plot(stos, sa_times, marker='o', label="SA (straggle)")
axs[0,0].axhline(su_natural_time, linestyle='--', color='green', label="SU (natural)")
axs[0,0].axhline(su_straggle_time, linestyle='--', color='red', label="SU (straggle)")
axs[0,0].set_xlabel("STO")
axs[0,0].set_ylabel("Epoch Time (s)")
axs[0,0].set_title("Epoch Time vs STO")
# micro_step_time
sa_micro = [data["time"]["sa"]["straggle"][sto]["epochs"]["0"]["micro_step_time"] for sto in stos]
su_natural_micro  = data["time"]["su"]["natural"]["epochs"]["0"]["micro_step_time"]
su_straggle_micro = data["time"]["su"]["straggle"]["epochs"]["0"]["micro_step_time"]
axs[0,1].plot(stos, sa_micro, marker='o', label="SA (straggle)")
axs[0,1].axhline(su_natural_micro, linestyle='--', color='green', label="SU (natural)")
axs[0,1].axhline(su_straggle_micro, linestyle='--', color='red', label="SU (straggle)")
axs[0,1].set_xlabel("STO")
axs[0,1].set_ylabel("Micro Step Time (s)")
axs[0,1].set_title("Micro Step Time vs STO")
# step_time
sa_step = [data["time"]["sa"]["straggle"][sto]["epochs"]["0"]["step_time"] for sto in stos]
su_natural_step  = data["time"]["su"]["natural"]["epochs"]["0"]["step_time"]
su_straggle_step = data["time"]["su"]["straggle"]["epochs"]["0"]["step_time"]
axs[0,2].plot(stos, sa_step, marker='o', label="SA (straggle)")
axs[0,2].axhline(su_natural_step, linestyle='--', color='green', label="SU (natural)")
axs[0,2].axhline(su_straggle_step, linestyle='--', color='red', label="SU (straggle)")
axs[0,2].set_xlabel("STO")
axs[0,2].set_ylabel("Step Time (s)")
axs[0,2].set_title("Step Time vs STO")

axs[0,0].axhline(data["time"]["sa"]["natural"][25]["epochs"]["0"]["epoch_time"], linestyle="--", color='orange', label="SA (natural)")
axs[0,1].axhline(data["time"]["sa"]["natural"][25]["epochs"]["0"]["micro_step_time"], linestyle="--", color='orange', label="SA (natural)")
axs[0,2].axhline(data["time"]["sa"]["natural"][25]["epochs"]["0"]["step_time"], linestyle="--", color='orange', label="SA (natural)")

axs[0,0].legend()
axs[0,1].legend()
axs[0,2].legend()

######################################


labels = []
res_n_vals = []
res_k_vals = []

for key in ["natural", "straggle"]:
    labels.append(f"SU {key}")
    res_n_vals.append(data["packets"]["su"][key]["res_n"])
    res_k_vals.append(data["packets"]["su"][key]["res_k"])

labels.append("SA nat 25")
res_n_vals.append(data["packets"]["sa"]["natural"][25]["res_n"])
res_k_vals.append(data["packets"]["sa"]["natural"][25]["res_k"])

for sto in [5, 25, 50, 75, 100, 125]:
    labels.append(f"SA str {sto}")
    res_n_vals.append(data["packets"]["sa"]["straggle"][sto]["res_n"])
    res_k_vals.append(data["packets"]["sa"]["straggle"][sto]["res_k"])

totals = [n + k for n, k in zip(res_n_vals, res_k_vals)]
res_n_pct = [n / t * 100 if t > 0 else 0 for n, t in zip(res_n_vals, totals)]
res_k_pct = [k / t * 100 if t > 0 else 0 for k, t in zip(res_k_vals, totals)]

x = np.arange(len(labels))
bars_n = axs[1,0].bar(x, res_n_pct, label="res_n")
bars_k = axs[1,0].bar(x, res_k_pct, bottom=res_n_pct, label="res_k")
axs[1,0].set_xticks(x)
axs[1,0].set_xticklabels(labels, rotation=45, ha='right')
axs[1,0].set_ylabel("Packets (%)")
axs[1,0].set_title("res_n / res_k")
axs[1,0].legend()

for i, (n, k) in enumerate(zip(res_n_pct, res_k_pct)):
    if n > 0 and n < 5:
       axs[1,0].annotate(f"{n:.2f}%", (x[i], n / 2), ha='center', fontsize=7, va='center', xytext=(0, 15), textcoords='offset points',arrowprops=dict(arrowstyle='->', lw=0.5))
    elif n >= 5:
      axs[1,0].text(x[i], n / 2, f"{n:.1f}%", ha='center', va='center', fontsize=7)
    if k > 0 and k < 5:
      axs[1,0].annotate(f"{k:.2f}%", (x[i], n + k / 2), ha='center', fontsize=7, va='center', xytext=(0, -15), textcoords='offset points',arrowprops=dict(arrowstyle='->', lw=0.5))
    elif k >= 5:
      axs[1,0].text(x[i], n + k / 2, f"{k:.1f}%", ha='center', va='center', fontsize=7)



labels = []
re_vals = []
re_to_vals = []
re_out_vals = []

for key in ["natural", "straggle"]:
    labels.append(f"SU {key}")
    d = data["packets"]["su"][key]
    re_vals.append(d["re"])
    re_to_vals.append(d["re_to"])
    re_out_vals.append(d["re_out"])

labels.append("SA nat 25")
d = data["packets"]["sa"]["natural"][25]
re_vals.append(d["re"])
re_to_vals.append(d["re_to"])
re_out_vals.append(d["re_out"])

for sto in [5, 25, 50, 75, 100, 125]:
    labels.append(f"SA str {sto}")
    d = data["packets"]["sa"]["straggle"][sto]
    re_vals.append(d["re"])
    re_to_vals.append(d["re_to"])
    re_out_vals.append(d["re_out"])

totals = [r + rt + ro for r, rt, ro in zip(re_vals, re_to_vals, re_out_vals)]
re_pct = [r / t * 100 if t > 0 else 0 for r, t in zip(re_vals, totals)]
re_to_pct = [rt / t * 100 if t > 0 else 0 for rt, t in zip(re_to_vals, totals)]
re_out_pct = [ro / t * 100 if t > 0 else 0 for ro, t in zip(re_out_vals, totals)]

x = np.arange(len(labels))
bottom_re_to = re_pct
bottom_re_out = [r + rt for r, rt in zip(re_pct, re_to_pct)]

axs[1,1].bar(x, re_pct, label="re")
axs[1,1].bar(x, re_to_pct, bottom=bottom_re_to, label="re_to")
axs[1,1].bar(x, re_out_pct, bottom=bottom_re_out, label="re_out")
axs[1,1].set_xticks(x)
axs[1,1].set_xticklabels(labels, rotation=45, ha='right')
axs[1,1].set_ylabel("Retransmissions (%)")
axs[1,1].set_title("re / re_to / re_out")
axs[1,1].legend()






labels = []
old_in_vals = []
old_out_vals = []
syn_in_vals = []
syn_out_vals = []

for key in ["natural", "straggle"]:
    labels.append(f"SU {key}")
    d = data["packets"]["su"][key]
    old_in_vals.append(d["old_in"])
    old_out_vals.append(d["old_out"])
    syn_in_vals.append(d["syn_in"])
    syn_out_vals.append(d["syn_out"])

labels.append("SA nat 25")
d = data["packets"]["sa"]["natural"][25]
old_in_vals.append(d["old_in"])
old_out_vals.append(d["old_out"])
syn_in_vals.append(d["syn_in"])
syn_out_vals.append(d["syn_out"])

for sto in [5, 25, 50, 75, 100, 125]:
    labels.append(f"SA str {sto}")
    d = data["packets"]["sa"]["straggle"][sto]
    old_in_vals.append(d["old_in"])
    old_out_vals.append(d["old_out"])
    syn_in_vals.append(d["syn_in"])
    syn_out_vals.append(d["syn_out"])

x = np.arange(len(labels))
width = 0.35

# old_in / old_out stack
axs[1,2].bar(x - width/2, old_in_vals, width, label="old_in")
axs[1,2].bar(x - width/2, old_out_vals, width, bottom=old_in_vals, label="old_out")

# syn_in / syn_out stack
axs[1,2].bar(x + width/2, syn_in_vals, width, label="syn_in")
axs[1,2].bar(x + width/2, syn_out_vals, width, bottom=syn_in_vals, label="syn_out")

axs[1,2].set_xticks(x)
axs[1,2].set_xticklabels(labels, rotation=45, ha='right')
axs[1,2].set_ylabel("Packets")
axs[1,2].set_title("old / syn")
axs[1,2].legend()

# Inset zoomed into syn range
axins = axs[1,2].inset_axes([0.5, 0.45, 0.45, 0.45])
syn_width = width / 2
axins.bar(x - syn_width/2, syn_in_vals, syn_width, color='C2', label="syn_in")
axins.bar(x + syn_width/2, syn_out_vals, syn_width, color='C3', label="syn_out")
axins.set_yscale('log')
axins.set_xticks(x)
axins.set_xticklabels(labels, rotation=45, ha='right', fontsize=5)
axins.set_title("syn (zoomed)", fontsize=7)

fig.tight_layout()
fig.savefig(script_dir / "plots.pdf", dpi=300, bbox_inches='tight')
plt.show()
