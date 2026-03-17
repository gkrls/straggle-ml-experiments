
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

data = {
  "time" : {},
  "packets" : {
    "sa" : {
      "natural-new": {
        25: {"res_n":80771787,  "res_k":3447498603,"ok":2962568085,"ok_to":10158211,"re":6717,     "re_to":381097887,"re_out":13524243,"old_in":99981779,"old_out":14531631+77498142,"syn_in":8734834,"syn_out":7498380},
        50: {"res_n":110118300, "res_k":3418792140,"ok":2970759400,"ok_to":6094678, "re":207375154,"re_to":233932463,"re_out":13519116,"old_in":97732887,"old_out":13510556+76919517,"syn_in":8397073,"syn_out":7039892},
        75: {"res_n":3291855594,"res_k":243900008, "ok":3498839767,"ok_to":311398,  "re":456513102,"re_to":6151765,  "re_out":4361306, "old_in":5038665, "old_out":981236  +4026871, "syn_in":973915, "syn_out":653314 },
        100:{"res_n":3528715008,"res_k":7069344,   "ok":3534720163,"ok_to":10787,   "re":476269580,"re_to":1780668,  "re_out":378973,  "old_in":81980,   "old_out":13819   +68161,   "syn_in":54890,  "syn_out":36178  },
        125:{"res_n":3535611390,"res_k":172962,    "ok":3535750915,"ok_to":4610,    "re":466192513,"re_to":1699391,  "re_out":304784,  "old_in":2135,    "old_out":540     +1595,    "syn_in":425,    "syn_out":414    },
        150:{"res_n":3535527216,"res_k":257136,    "ok":3535752703,"ok_to":4804,    "re":469941086,"re_to":1569455,  "re_out":304153,  "old_in":2168,    "old_out":518     +1650,    "syn_in":1532,   "syn_out":953    },
      },

      "natural" : {
        25: {"res_n":29620679,  "res_k":3256481426,"ok":2737060586,"ok_to":10807891,"re":6437,     "re_to":287248245,"re_out":3237954, "old_in":38641711,"old_out":29894074+6589020, "syn_in":5410083,"syn_out":4634305},
        50: {"res_n":117442345, "res_k":3411674491,"ok":2972481017,"ok_to":6146444, "re":180937656,"re_to":206530700,"re_out":13679163,"old_in":95499656,"old_out":73620084+15010294,"syn_in":9103010,"syn_out":7511836},
        75: {"res_n":3288394853,"res_k":247365709, "ok":3498292457,"ok_to":417500,  "re":421136021,"re_to":10056750, "re_out":4517330, "old_in":5389566, "old_out":1102249 +4262801, "syn_in":1039386,"syn_out":692470 },
        100:{"res_n":3525564132,"res_k":10220220,  "ok":3534273602,"ok_to":26475,   "re":470019902,"re_to":1773258,  "re_out":531711,  "old_in":157504,  "old_out":31911   +125593,  "syn_in":81165,  "syn_out":52197},
        125:{"res_n":3535611390,"res_k":172962,    "ok":3535750915,"ok_to":4610,    "re":466192513,"re_to":1699391,  "re_out":304784,  "old_in":2135,    "old_out":540     +1595,    "syn_in":425,    "syn_out":414},
        150:{"res_n":3535527216,"res_k":257136,    "ok":3535752703,"ok_to":4804,    "re":469941086,"re_to":1569455,  "re_out":304153,  "old_in":2168,    "old_out":518     +1650,    "syn_in":1532,   "syn_out":953},
      },
      "straggle": {
         5: {"res_n":0,         "res_k":3535756854,"ok":2929216156,"ok_to":17272103,"re":5325,     "re_to":430582943,"re_out":2147772, "old_in":4866518,"old_out":4581392+257143,    "syn_in":15486,  "syn_out":14362},
        25: {"res_n":0,         "res_k":3535784352,"ok":2940183144,"ok_to":6304425,"re":5273,      "re_to":517810229,"re_out":6658521, "old_in":5985792,"old_out":5772662+213130,    "syn_in":1920,   "syn_out":1641},
        50: {"res_n":0,         "res_k":3535783756,"ok":2940566423,"ok_to":5928440,"re":201256633, "re_to":206530747,"re_out":1746410, "old_in":6010247,"old_out":5767938+241594,    "syn_in":2761,   "syn_out":2241},
        75: {"res_n":60,        "res_k":3535770868,"ok":2946064098,"ok_to":425746, "re":655953794, "re_to":30944135, "re_out":9711786, "old_in":6104957,"old_out":5830463+261189,    "syn_in":10745,  "syn_out":9607},
        100:{"res_n":2918862,   "res_k":3532865490,"ok":2946668233,"ok_to":305762, "re":1109214743,"re_to":22413261, "re_out":9195524, "old_in":5985412,"old_out":5681476+303936,    "syn_in":1404,   "syn_out":1404},
        125:{"res_n":2918862,   "res_k":3532865490,"ok":2946968060,"ok_to":5935,   "re":1532690720,"re_to":23764888, "re_out":15993296,"old_in":5985408,"old_out":392623+5592785,    "syn_in":1472,   "syn_out":1472}
      }
    },
    "su" : {
      "natural" : {"res_n":3535784352,"res_k":0,"ok":3520818668,"ok_to":14965684,"re":7728,"re_to":455257781, "re_out":357053,"old_in":0,"old_out":0,"syn_in":0,"syn_out":0},
      "straggle": {"res_n":3535784352,"res_k":0,"ok":3528580494,"ok_to":7203858, "re":2455,"re_to":7625910118,"re_out":302504,"old_in":0,"old_out":0,"syn_in":0,"syn_out":0},
    },
  }
}

script_dir = Path(__file__).parent
stos = [5,25,50,75,100,125,150]

# ------------------ LOAD SA ------------------

for sto in stos:
    paths = {
        "straggle":    script_dir / "sa"             / f"gpt2_ga5_1epoch_sa_{sto}-1_straggl_3.15.50-200.json",
        "natural":     script_dir / "sa-natural"     / f"gpt2_ga5_1epoch_sa_{sto}-1_natural.json",
        "natural-new": script_dir / "sa-new-natural" / f"gpt2_ga5_1epoch_sa_{sto}-1_natural.json",
    }

    for key, path in paths.items():
        if path.exists():
            with open(path) as f:
                data["time"].setdefault("sa", {}).setdefault(key, {})[sto] = json.load(f)

# ------------------ LOAD SU ------------------

with open(script_dir / "su" / "gpt2_ga5_1epoch_su_natural.json") as f:
    data["time"].setdefault("su", {})["natural"] = json.load(f)

with open(script_dir / "su" / "gpt2_ga5_1epoch_su_straggl_3.15.50-200.json") as f:
    data["time"].setdefault("su", {})["straggle"] = json.load(f)

# ------------------ PLOTS ------------------

fig, axs = plt.subplots(2, 4, figsize=(18,10))

# ---------- TIME METRICS ----------
metrics = [
    ("epoch_time", axs[0,0], "Epoch Time (s)"),
    ("micro_step_time", axs[0,1], "Micro Step Time (s)"),
    ("step_time", axs[0,2], "Step Time (s)"),
]

vers = ["straggle", "natural", "natural-new"]
cfgs = [{'color': 'blue', 'marker': 'o'}, {'color': 'green', 'marker': 'x'}, {'color': 'orange', 'marker': '^'}]
for metric, ax, ylabel in metrics:
    for ver,cfg in zip(vers, cfgs):
        if ver not in data["time"]["sa"]:
            continue
        x = sorted(data["time"]["sa"][ver].keys())
        y = [data["time"]["sa"][ver][s]["epochs"]["0"][metric] for s in x]
        ax.plot(x, y, label=f"SA ({ver})", **cfg)

    ax.axhline(data["time"]["su"]["natural"]["epochs"]["0"][metric], color="black", linestyle='--', label="SU (natural)")
    ax.axhline(data["time"]["su"]["straggle"]["epochs"]["0"][metric], color="red", linestyle='--', label="SU (straggle)")

    ax.set_xlabel("STO")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs STO")
    ax.legend()

# ---------- res_n / res_k (SIDE-BY-SIDE natural vs natural-new) ----------
ax = axs[1,0]

labels = []
res_n = []
res_k = []

# ---- SU ----
for key in ["natural", "straggle"]:
    d = data["packets"]["su"][key]
    labels.append(f"SU {key}")
    res_n.append(d["res_n"])
    res_k.append(d["res_k"])

# ---- SA natural + natural-new (side-by-side) ----
for sto in stos:
    if sto in data["packets"]["sa"]["natural"]:
        d = data["packets"]["sa"]["natural"][sto]
        labels.append(f"SA nat-1 {sto}")
        res_n.append(d["res_n"])
        res_k.append(d["res_k"])

    if sto in data["packets"]["sa"]["natural-new"]:
        d = data["packets"]["sa"]["natural-new"][sto]
        labels.append(f"SA nat-2 {sto}")
        res_n.append(d["res_n"])
        res_k.append(d["res_k"])

# ---- SA straggle ----
for sto in stos:
    if sto in data["packets"]["sa"]["straggle"]:
        d = data["packets"]["sa"]["straggle"][sto]
        labels.append(f"SA str {sto}")
        res_n.append(d["res_n"])
        res_k.append(d["res_k"])

tot = np.array(res_n) + np.array(res_k)
res_n_pct = np.divide(res_n, tot, where=tot>0) * 100
res_k_pct = np.divide(res_k, tot, where=tot>0) * 100

x = np.arange(len(labels))

ax.bar(x, res_n_pct, label="res_n")
ax.bar(x, res_k_pct, bottom=res_n_pct, label="res_k")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel("Packets (%)")
ax.set_title("res_n / res_k")
ax.legend()

# ---------- retransmissions ----------
ax = axs[1,1]

labels = []
re_vals = []
re_to_vals = []
re_out_vals = []

# ---- SU ----
for key in ["natural", "straggle"]:
    labels.append(f"SU {key}")
    d = data["packets"]["su"][key]
    re_vals.append(d["re"])
    re_to_vals.append(d["re_to"])
    re_out_vals.append(d["re_out"])

# ---- SA ----
for sto in stos:
    if sto in data["packets"]["sa"]["natural"]:
        labels.append(f"SA nat {sto}")
        d = data["packets"]["sa"]["natural"][sto]
        re_vals.append(d["re"])
        re_to_vals.append(d["re_to"])
        re_out_vals.append(d["re_out"])

    if sto in data["packets"]["sa"]["natural-new"]:
        labels.append(f"SA new {sto}")
        d = data["packets"]["sa"]["natural-new"][sto]
        re_vals.append(d["re"])
        re_to_vals.append(d["re_to"])
        re_out_vals.append(d["re_out"])

for sto in stos:
    if sto in data["packets"]["sa"]["straggle"]:
        labels.append(f"SA str {sto}")
        d = data["packets"]["sa"]["straggle"][sto]
        re_vals.append(d["re"])
        re_to_vals.append(d["re_to"])
        re_out_vals.append(d["re_out"])

# ---- normalize toggle ----
normalize = True

tot = np.array(re_vals) + np.array(re_to_vals) + np.array(re_out_vals)
re_plot     = (np.divide(re_vals, tot, where=tot>0) * 100) if normalize else np.array(re_vals)
re_to_plot  = (np.divide(re_to_vals, tot, where=tot>0) * 100) if normalize else np.array(re_to_vals)
re_out_plot = (np.divide(re_out_vals, tot, where=tot>0) * 100) if normalize else np.array(re_out_vals)
ylabel = "Retransmissions (%)" if normalize else "Retransmissions"
if normalize: ax.set_ylim(0, 100)

# ---- plot ----
x = np.arange(len(labels))

ax.bar(x, re_plot, label="re")
ax.bar(x, re_to_plot, bottom=re_plot, label="re_to")
ax.bar(x, re_out_plot, bottom=re_plot + re_to_plot, label="re_out")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel(ylabel)
ax.set_title("re / re_to / re_out")
ax.legend()

# ----- ok / ok_to ------
ax = axs[1,2]
labels, ok_vals, ok_to_vals = [], [], []

# ---- SU ----
for key in ["natural", "straggle"]:
    labels.append(f"SU {key}")
    d = data["packets"]["su"][key]
    ok_vals.append(d["ok"])
    ok_to_vals.append(d["ok_to"])

# ---- SA ----
for sto in stos:
    if sto in data["packets"]["sa"]["natural"]:
        labels.append(f"SA nat {sto}")
        d = data["packets"]["sa"]["natural"][sto]
        ok_vals.append(d["ok"])
        ok_to_vals.append(d["ok_to"])

    if sto in data["packets"]["sa"]["natural-new"]:
        labels.append(f"SA new {sto}")
        d = data["packets"]["sa"]["natural-new"][sto]
        ok_vals.append(d["ok"])
        ok_to_vals.append(d["ok_to"])

for sto in stos:
    if sto in data["packets"]["sa"]["straggle"]:
        labels.append(f"SA str {sto}")
        d = data["packets"]["sa"]["straggle"][sto]
        ok_vals.append(d["ok"])
        ok_to_vals.append(d["ok_to"])

normalize  = True
tot        = np.array(ok_vals) + np.array(ok_to_vals)
ok_plot    = (np.divide(ok_vals, tot, where=tot>0) * 100) if normalize else np.array(ok_vals)
ok_to_plot = (np.divide(ok_to_vals, tot, where=tot>0) * 100) if normalize else np.array(ok_to_vals)
ylabel     = "Packets (%)" if normalize else "Packets"
if normalize: ax.set_ylim(0, 100)
x = np.arange(len(labels))
ax.bar(x, ok_plot, label="ok")
ax.bar(x, ok_to_plot, bottom=ok_plot, label="ok_to")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel(ylabel)
ax.set_title("ok / ok_to")
ax.legend()

# ---------- old / syn ----------
ax = axs[1,3]
labels, old_in_vals, old_out_vals, syn_in_vals, syn_out_vals = [], [], [], [], []

# ---- SU ----
for key in ["natural", "straggle"]:
    labels.append(f"SU {key}")
    d = data["packets"]["su"][key]
    old_in_vals.append(d["old_in"])
    old_out_vals.append(d["old_out"])
    syn_in_vals.append(d["syn_in"])
    syn_out_vals.append(d["syn_out"])

# ---- SA ----
for sto in stos:
    if sto in data["packets"]["sa"]["natural"]:
        labels.append(f"SA nat-1 {sto}")
        d = data["packets"]["sa"]["natural"][sto]
        old_in_vals.append(d["old_in"])
        old_out_vals.append(d["old_out"])
        syn_in_vals.append(d["syn_in"])
        syn_out_vals.append(d["syn_out"])

    if sto in data["packets"]["sa"]["natural-new"]:
        labels.append(f"SA nat-2 {sto}")
        d = data["packets"]["sa"]["natural-new"][sto]
        old_in_vals.append(d["old_in"])
        old_out_vals.append(d["old_out"])
        syn_in_vals.append(d["syn_in"])
        syn_out_vals.append(d["syn_out"])

for sto in stos:
    if sto in data["packets"]["sa"]["straggle"]:
        labels.append(f"SA str {sto}")
        d = data["packets"]["sa"]["straggle"][sto]
        old_in_vals.append(d["old_in"])
        old_out_vals.append(d["old_out"])
        syn_in_vals.append(d["syn_in"])
        syn_out_vals.append(d["syn_out"])

# ---- plot (unchanged spacing logic) ----
x = np.arange(len(labels))
w = 0.35

ax.set_xticks(x)
ax.set_ylabel("Packets")
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.bar(x - w/2, old_in_vals, w, label="old_in")
ax.bar(x - w/2, old_out_vals, w, bottom=old_in_vals, label="old_out")
ax.bar(x + w/2, syn_in_vals, w, label="syn_in")
ax.bar(x + w/2, syn_out_vals, w, bottom=syn_in_vals, label="syn_out")
ax.legend()

#INSET
start = labels.index("SA nat-1 75") if "SA nat-1 75" in labels else 0
end   = labels.index("SA str 5") + 1 if "SA str 5" in labels else len(labels)
axins = ax.inset_axes([0.5, 0.45, 0.45, 0.45])
xs = x[start:end]

# ---- old stack (left) ----
axins.bar(xs - w/2, old_in_vals[start:end], w)
axins.bar(xs - w/2, old_out_vals[start:end], w, bottom=old_in_vals[start:end])
# ---- syn stack (right) ----
axins.bar(xs + w/2, syn_in_vals[start:end], w)
axins.bar(xs + w/2, syn_out_vals[start:end], w, bottom=syn_in_vals[start:end])
axins.set_yscale('log')
axins.set_ylabel("Packets (log)")
axins.set_xticks(xs)
axins.set_xticklabels(labels[start:end], rotation=45, ha='right', fontsize=8)
axins.set_title("old / syn (zoomed)", fontsize=7)

fig.tight_layout()
fig.savefig(script_dir / "plots.pdf", dpi=300, bbox_inches='tight')
plt.show()