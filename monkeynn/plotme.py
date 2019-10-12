import pandas as pd
import plotly.express as px

x = [[14, 38, 22],
     [68, 14, 90],
     [60, 36, 66],
     [92, 37, 74],
     [94, 84, 44],
     [87, 45, 20],
     [52, 77, 13],
     [27, 44,  1],
     [43, 54, 31],
     [66,  4, 66],
     [38, 69, 65],
     [88, 92,  5],
     [71, 11, 51],
     [41, 35, 33],
     [14,  8, 94],
     [80, 17, 99],
     [2, 36, 13],
     [13, 93, 34],
     [43, 40, 52],
     [74, 27, 54]]
pdf = pd.DataFrame(x, columns=['x', 'y', 'z'])
fig = px.scatter_3d(pdf)
fig.show()