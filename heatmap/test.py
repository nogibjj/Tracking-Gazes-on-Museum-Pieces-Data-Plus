import matplotlib
from matplotlib import cm
import numpy as np

data = {'A': 0, 'B': 2, 'C': 4, 'D': 1, 'E': 3, 'F': 5}
values = list(data.values())

norm = matplotlib.colors.Normalize(vmin=min(values), vmax=max(values), clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greys_r)
rgba_colors = mapper.to_rgba(values)
node_color = {key: tuple(np.array(color[:3]) * 255) for key, color in zip(data.keys(), rgba_colors)}

print(node_color)