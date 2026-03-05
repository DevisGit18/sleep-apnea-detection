import os, argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

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

def bandpass_filter(sig, fs, lo=0.17, hi=0.4, order=4):
    nyq  = fs / 2.0
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, sig)

def get_label(win_start, win_end, events_df):
    duration = (win_end - win_start).total_seconds()
    for _, ev in events_df.iterrows():
        overlap = (min(win_end, ev['end']) - max(win_start, ev['start'])).total_seconds()
        if overlap > 0.5 * duration:
            return ev['event']
    return 'Normal'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir',  required=True)
    parser.add_argument('-out_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []

    for pid in sorted(os.listdir(args.in_dir)):
        folder = os.path.join(args.in_dir, pid)
        if not os.path.isdir(folder):
            continue
        print(f"processing {pid}...")

        flow_f   = find_file(folder, ['flow'],   exclude=['events'])
        thorac_f = find_file(folder, ['thorac'])
        spo2_f   = find_file(folder, ['spo2'])
        events_f = find_file(folder, ['flow', 'events'])

        if not all([flow_f, thorac_f, spo2_f, events_f]):
            print(f"  missing files, skipping")
            continue

        flow   = parse_signal(flow_f)
        thorac = parse_signal(thorac_f)
        spo2   = parse_signal(spo2_f)
        events = parse_events(events_f)

        flow_f2   = bandpass_filter(flow.values,   fs=32)
        thorac_f2 = bandpass_filter(thorac.values, fs=32)
        spo2_f2   = bandpass_filter(spo2.values,   fs=4)

        win, step, n = 960, 480, 0
        for i in range(0, len(flow_f2) - win + 1, step):
            ws = flow.index[i]
            we = flow.index[i + win - 1]
            si = i // 8
            se = si + 120
            if se > len(spo2_f2):
                break
            label = get_label(ws, we, events)
            row = {'participant': pid, 'win_start': str(ws), 'label': label}
            for j, v in enumerate(flow_f2[i:i+win]):   row[f'flow_{j}']   = v
            for j, v in enumerate(thorac_f2[i:i+win]): row[f'thorac_{j}'] = v
            for j, v in enumerate(spo2_f2[si:se]):     row[f'spo2_{j}']   = v
            all_rows.append(row)
            n += 1
        print(f"  {pid}: {n} windows")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.out_dir, 'breathing_dataset.csv'), index=False)
    print(f"saved {len(df)} windows to {args.out_dir}/breathing_dataset.csv")