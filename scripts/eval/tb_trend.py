import sys, glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
TAGS = {'apex cm': ('Episode_Metrics/poultry_swing_apex_m', 100), 'clear cm': ('Episode_Metrics/poultry_swing_clearance_m', 100),
        'plow': ('Episode_Metrics/poultry_plow_cost', 1000), 'lowtrv': ('Episode_Metrics/poultry_toe_low_travel_mps', 100),
        'slide': ('Episode_Metrics/poultry_toe_contact_slide_mps', 100), 'toedn': ('Episode_Metrics/poultry_liftoff_toe_down_deg', 1),
        'drag N': ('Episode_Metrics/poultry_drag_force_n', 1), 'layer': ('Episode_Metrics/poultry_layer_height_m', 100),
        'R_clr': ('Episode_Reward/poultry_clearance', 100), 'R_plow': ('Episode_Reward/poultry_plow', 100),
        'R_swing': ('Episode_Reward/poultry_swing_height', 100), 'R_level': ('Episode_Reward/poultry_foot_level', 100),
        'terrain': ('Curriculum/terrain_levels', 1), 'falls': ('Episode_Termination/fell_over', 1)}
ea = EventAccumulator(glob.glob(sys.argv[1] + '/events*')[0], size_guidance={'scalars': 0}); ea.Reload()
step = int(sys.argv[2]) if len(sys.argv) > 2 else 500
data = {}
for k, (t, s) in TAGS.items():
    if t in ea.Tags()['scalars']:
        data[k] = {e.step: e.value * s for e in ea.Scalars(t)}
steps = sorted(next(iter(data.values())))
lo, hi = steps[0], steps[-1]
print('window', ' '.join(f'{k:>8}' for k in data))
b = lo
while b <= hi:
    w = [s for s in steps if b <= s < b + step]
    if w:
        print(f'{w[0]:>6}', ' '.join(f'{np.mean([data[k][s] for s in w if s in data[k]]):8.3f}' for k in data))
    b += step
