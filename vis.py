import os, argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def find_file(folder, keywords, exclude=None):
    for fname in os.listdir(folder):
        f = fname.lower()
        if all(k.lower() in f for k in keywords):
            if exclude and any(e.lower() in f for e in exclude):
                continue
            return os.path.join(folder, fname)
    return None

def parse_signal(filepath):
    timestamps, values = [], []
    reading = False
    with open(filepath, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.lower() == 'data:':
                reading = True
                continue
            if reading and ';' in line:
                parts = line.split(';')
                try:
                    ts  = pd.to_datetime(parts[0].strip(), format='%d.%m.%Y %H:%M:%S,%f')
                    val = float(parts[1].strip())
                    timestamps.append(ts)
                    values.append(val)
                except:
                    pass
    return pd.Series(values, index=pd.DatetimeIndex(timestamps), name='value')

def parse_events(filepath):
    events = []
    with open(filepath, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if ';' not in line or '-' not in line.split(';')[0]:
                continue
            try:
                parts      = line.split(';')
                time_range = parts[0].strip()
                event_type = parts[2].strip()
                date_str   = time_range[:10]
                start_str, end_str = time_range[11:].split('-')
                start = pd.to_datetime(date_str + ' ' + start_str, format='%d.%m.%Y %H:%M:%S,%f')
                end   = pd.to_datetime(date_str + ' ' + end_str,   format='%d.%m.%Y %H:%M:%S,%f')
                if end < start:
                    end += pd.Timedelta(days=1)
                events.append({'start': start, 'end': end, 'event': event_type})
            except:
                pass
    return pd.DataFrame(events)

def plot_participant(folder, out_dir):
    flow_f   = find_file(folder, ['flow'],   exclude=['events'])
    thorac_f = find_file(folder, ['thorac'])
    spo2_f   = find_file(folder, ['spo2'])
    events_f = find_file(folder, ['flow', 'events'])

    if not all([flow_f, thorac_f, spo2_f, events_f]):
        print(f"missing files in {folder}, skipping")
        return

    flow   = parse_signal(flow_f)
    thorac = parse_signal(thorac_f)
    spo2   = parse_signal(spo2_f)
    events = parse_events(events_f)
    pid    = os.path.basename(folder)

    flow_ds   = flow.iloc[::32]
    thorac_ds = thorac.iloc[::32]
    spo2_ds   = spo2.iloc[::4]

    colors = {
        'Hypopnea':          '#FF8C00',
        'Obstructive Apnea': '#CC0000',
        'Central Apnea':     '#9900CC',
        'Mixed Apnea':       '#0066CC',
    }

    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
    fig.suptitle(f'Sleep Study - Participant {pid}', fontsize=16, fontweight='bold', y=0.98)

    for sig, label, color, ax in [
        (flow_ds,   'Nasal Airflow',     'steelblue', axes[0]),
        (thorac_ds, 'Thoracic Movement', 'seagreen',  axes[1]),
        (spo2_ds,   'SpO2 (%)',          'crimson',   axes[2]),
    ]:
        t0  = sig.index[0]
        hrs = (sig.index - t0).total_seconds() / 3600.0
        ax.plot(hrs, sig.values, color=color, linewidth=0.4, alpha=0.85)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        for _, ev in events.iterrows():
            s = (ev['start'] - t0).total_seconds() / 3600.0
            e = (ev['end']   - t0).total_seconds() / 3600.0
            ax.axvspan(s, e, alpha=0.35, color=colors.get(ev['event'], '#888'), linewidth=0)

    axes[2].set_xlabel('Time (hours from start)', fontsize=11)
    patches = [mpatches.Patch(color=c, label=ev, alpha=0.6) for ev, c in colors.items()]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=10,
               title='Breathing Events', title_fontsize=10, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'{pid}_visualization.pdf')
    fig.savefig(out, format='pdf', dpi=150, bbox_inches='tight')
    print(f"saved → {out}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', required=True, help='path to participant folder e.g. Data/AP01')
    args    = parser.parse_args()
    folder  = args.name
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(folder))), 'Visualizations')
    plot_participant(folder, out_dir)