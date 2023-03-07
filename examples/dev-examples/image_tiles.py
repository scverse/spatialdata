##
# from https://gist.github.com/kevinyamauchi/77f986889b7626db4ab3c1075a3a3e5e

import numpy as np
from matplotlib import pyplot as plt
from skimage import draw

import spatialdata as sd
from spatialdata._dl.datasets import ImageTilesDataset

##
coordinates = np.array([[10, 10], [20, 20], [50, 30], [90, 70], [20, 80]])

radius = 5

colors = np.array(
    [
        [102, 194, 165],
        [252, 141, 98],
        [141, 160, 203],
        [231, 138, 195],
        [166, 216, 84],
    ]
)

##
# make an image with spots
image = np.zeros((100, 100, 3), dtype=np.uint8)

for spot_color, centroid in zip(colors, coordinates):
    rr, cc = draw.disk(centroid, radius=radius)

    for color_index in range(3):
        channel_dims = color_index * np.ones((len(rr),), dtype=int)
        image[rr, cc, channel_dims] = spot_color[color_index]

# plt.imshow(image)
# plt.show()

##
sd_image = sd.Image2DModel.parse(image, dims=("y", "x", "c"))

# circles coordinates are xy, so we flip them here.
circles = sd.ShapesModel.parse(coordinates[:, [1, 0]], radius=radius, geometry=0)
sdata = sd.SpatialData(images={"image": sd_image}, shapes={"spots": circles})
sdata

##
ds = ImageTilesDataset(
    sdata=sdata,
    regions_to_images={"/shapes/spots": "/images/image"},
    tile_dim_in_units=10,
    tile_dim_in_pixels=32,
    target_coordinate_system="global",
    data_dict_transform=None,
)

print(f"this dataset as {len(ds)} items")

##
# we can use the __getitem__ interface to get one of the sample crops
print(ds[0])


##
# now we plot all of the crops
def plot_sdata_dataset(ds: ImageTilesDataset) -> None:
    n_samples = len(ds)
    fig, axs = plt.subplots(1, n_samples)

    for i, (image, region, index) in enumerate(ds):
        axs[i].imshow(image.transpose("y", "x", "c"))
        axs[i].set_title(f"region: {region}, index: {index}")
    plt.show()


plot_sdata_dataset(ds)

# TODO: code to be restored when the transforms will use the bounding box query
# ##
# # we can also use transforms to automatically extract the relevant data
# # into a datadictionary
#
# # map the SpatialData path to a data dict key
# data_mapping = {"images/image": "image"}
#
# # make the transform
# ds_transform = ImageTilesDataset(
#     sdata=sdata,
#     spots_element_key="spots",
#     transform=SpatialDataToDataDict(data_mapping=data_mapping),
# )
#
# print(f"this dataset as {len(ds_transform)} items")
#
# ##
# # now the samples are a dictionary with key "image" and the item is the
# # image array
# # this is useful because it is the expected format for many of the
# #
# ds_transform[0]
#
# ##
# # plot of each sample in the dataset
# plot_sdata_dataset(ds_transform)
