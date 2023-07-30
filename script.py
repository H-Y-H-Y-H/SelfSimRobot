#
#
# import plotly.express as px
# df = px.data.iris()
# fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
#                     color='species')
# fig.show()
#
# # A 4th dimension of the data can be represented thanks to the color of the markers. Also, values from the `species` column are used below to assign symbols to markers.
#
#
# import plotly.express as px
# df = px.data.iris()
# fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
#                     color='petal_length', symbol='species')
# fig.show()
#
# # Style 3d scatter plot
#
# # It is possible to customize the style of the figure through the parameters of `px.scatter_3d` for some options, or by updating the traces or the layout of the figure through `fig.update`.
#
#
# import plotly.express as px
# df = px.data.iris()
# fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
#                     color='petal_length', size='petal_length', size_max=18,
#                     symbol='species', opacity=0.7)
#
# # tight layout
# fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#
#
# # 3D Scatter Plot with go.Scatter3d
#
# # Basic 3D Scatter Plot
#
# # If Plotly Express does not provide a good starting point, it is also possible to use [the more generic `go.Scatter3D` class from `plotly.graph_objects`](/python/graph-objects/).
# # Like the [2D scatter plot](https://plotly.com/python/line-and-scatter/) `go.Scatter`, `go.Scatter3d` plots individual data in three-dimensional space.
#
#
# import plotly.graph_objects as go
# import numpy as np

# # Helix equation
# t = np.linspace(0, 10, 50)
# x, y, z = np.cos(t), np.sin(t), t
#
# fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
#                                    mode='markers')])
# fig.show()
#
#
# # 3D Scatter Plot with Colorscaling and Marker Styling
#
#
# import plotly.graph_objects as go
# import numpy as np
#
# # Helix equation
# t = np.linspace(0, 20, 100)
# x, y, z = np.cos(t), np.sin(t), t
#
# fig = go.Figure(data=[go.Scatter3d(
#     x=x,
#     y=y,
#     z=z,
#     mode='markers',
#     marker=dict(
#         size=12,
#         color=z,                # set color to an array/list of desired values
#         colorscale='Viridis',   # choose a colorscale
#         opacity=0.8
#     )
# )])
#
# # tight layout
# fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
# fig.show()


# Reference

# See https://plotly.com/python/reference/#scatter3d for more information and chart attribute options!
import numpy as np
import torch

a = [[1,2,3, 4, 5, 6],
      [7,8,9,10,11,12],
      [13,14,15,16,17,18]]
b = [[0,1,2, 3, 4, 5],
      [1,2,3, 4, 5, 6],
      [0,1,2, 3, 4, 5]]
a = torch.from_numpy(np.asarray(a))
# b = torch.from_numpy(np.asarray(b))
#
# a = a.reshape(list(b.shape[:2])+[a.shape[-1]])

# print(a.shape,a[:,:,0])
# target_img = [[1,2,3, 4, 5, 6],
#               [7,8,9,10,11,12],
#               [13,14,15,16,17,18]]
# target_img = np.asarray(target_img).T
# target_img = target_img.flatten()
# print(target_img)

c = torch.cumprod(a, -1)
print(c)
c = torch.roll(c, 1, -1)
print(c)
c[..., 0] = 1.
print(c)