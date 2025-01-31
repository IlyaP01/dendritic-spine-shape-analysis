from itertools import product, chain
import sys
import traceback
import numpy as np
from skimage.filters import median
from skimage.segmentation import morphological_chan_vese
import networkx as nx
import napari
from napari import Viewer
from napari.layers import Image
from magicgui import magicgui, widgets


def get_weight(n1, n2, img, factor, dx, dy, dz):
    z1, x1, y1 = n1
    z2, x2, y2 = n2
    I = (0.5 * img[z1][x1][y1] + 0.5 * img[z2][x2][y2]) / 255
    dist = np.sqrt((z1 - z2)**2*dz**2 + (x1 - x2)**2*dx**2 + (y1 - y2)**2*dy**2)
    return max(-I * factor + dist, 0)


def get_edges(node, img, factor, dx, dy, dz):
    box_coords = list(product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
    box_coords.remove((0, 0, 0))
    edges = []
    d, h, w = img.shape
    z, x, y = node
    for coord in box_coords:
        k = z + coord[0]
        i = x + coord[1]
        j = y + coord[2]
        if 0 <= i and i < h and 0 <= j and j < w and 0 <= k < z:
            neib = (k, i, j)
            edge = (node, neib, get_weight(node, neib, img, factor, dx, dy, dz))
            edges.append(edge)
    return edges


def floodfill(img, path, size=5, factor=0.6):
    print("Start floodfill")
    flooded = []
    box_coords = list(product(range(-size, size + 1), range(-size, size + 1),  range(-size, size + 1)))
    box_coords.remove((0, 0, 0))
    z, h, w = img.shape
    filtered = median(img)
    filtered = img
    for p in path:
        flooded.append(p)
        for coord in box_coords:
            k = p[0] + coord[0]
            i = p[1] + coord[1]
            j = p[2] + coord[2]
            if 0 <= i and i < h and 0 <= j and j < w and 0 <= k < z:
                if filtered[k][i][j] >= filtered[p[0]][p[1]][p[2]] * factor:
                    flooded.append((k, i, j))
    print("Floodfill finished")
    return flooded


def fmm_floodfill(img, path, fmm_ext_factor, fmm_int_factor, radius):
    print("Start FMM floodfill")
    init_contour = np.zeros(img.shape)
    pipe = np.zeros(img.shape)
    box_coords = list(product(range(-radius, radius + 1), range(-radius, radius + 1),  range(-radius, radius + 1)))
    z, h, w = img.shape
    for p in path:
        init_contour[p[0]][p[1]][p[2]] = 1
        for coord in box_coords:
            k = p[0] + coord[0]
            i = p[1] + coord[1]
            j = p[2] + coord[2]
            if 0 <= i and i < h and 0 <= j and j < w and 0 <= k < z:
                pipe[k][i][j] = img[k][i][j]

    seg = morphological_chan_vese(pipe, num_iter=100, init_level_set=init_contour, 
        lambda1=fmm_ext_factor, lambda2=fmm_int_factor, smoothing=0)
    print("FMM floodfill Finished")
    return np.argwhere(seg>0)


def find_path(img, src, dst, factor, dx, dy, dz):
    print("Start finding path")
    z, h, w = img.shape
    edges = list(chain.from_iterable([get_edges(node, img, factor, dx, dy, dz) for node in product(range(z), range(h), range(w))]))
    print("Graph built")
    g = nx.Graph()
    g.add_weighted_edges_from(edges)
    print("Finidg shortest")
    path = nx.shortest_path(g, source=src, target=dst, weight='weight')
    print("Path found")
    return path


def get_reconnection_widget(img, binary):
    viewer = napari.view_image(img, name='Image', title='Necks reconnection')
    viewer.add_image(binary, name='Binary', colormap='green', opacity=0.2)
    viewer.add_image(binary, name='Saved result', colormap='green', opacity=0.2, visible=False)
    viewer.add_points(ndim=3, size=3)
    viewer.add_shapes(ndim=3, name='ROIs')

    @magicgui(
        call_button="Reconnect",
        x_scale={"widget_type": "FloatSlider", 'min': 0, 'max': 1, 'step':0.001},
        y_scale={"widget_type": "FloatSlider", 'min': 0, 'max': 1, 'step':0.001},
        z_scale={"widget_type": "FloatSlider", 'min': 0, 'max': 1, 'step':0.001},
        intensity_factor={"widget_type": "FloatSlider", 'min': 0, 'max': 3, 'step':0.1},
        floodfill_radius={"widget_type": "Slider", 'min': 0, 'max': 20, 'step':1},
        floodfill_factor={"widget_type": "FloatSlider", 'min': 0, 'max': 1, 'step':0.1},
        fmm_ext_factor={"widget_type": "FloatSlider", 'min': 1, 'max': 10, 'step':0.5},
        fmm_int_factor={"widget_type": "FloatSlider", 'min': 1, 'max': 10, 'step':0.5}
    )
    def widget_necks_segmentation(viewer: Viewer, source_binary: Image,
        x_scale=1,
        y_scale=1,
        z_scale=1,
        intensity_factor=1,
        floodfill_radius = 4,
        floodfill_factor = 0.8,
        use_fmm=False,
        fmm_ext_factor=2,
        fmm_int_factor=6                          
    ) -> Image:
        if not hasattr(widget_necks_segmentation, "counter"):
            widget_necks_segmentation.counter = 0  # it doesn't exist yet, so initialize it
        widget_necks_segmentation.counter += 1
        binary = source_binary.data
        img = viewer.layers['Image'].data
        result = np.copy(binary)
        if 'Points' not in viewer.layers:
            print('Add pairs of points for connection')
            return
        if 'ROIs' not in viewer.layers:
            print('Add rectangle roi for every pair of points')
            return
        points = viewer.layers['Points'].data
        rois =  viewer.layers['ROIs'].data
        if (len(points) <= 0 or len(points) % 2 != 0 or len(rois) != len(points) // 2):
            print("Enter pairs of points and rectangle rois")
        else:
            print("Start processing")
            i = 1
            for (p1, p2), roi in zip(zip(points[::2], points[1::2]), rois):
                print(f"Neck {i}")
                z1, y1, x1 = (round(t) for t in p1)
                z2, y2, x2 = (round(t) for t in p2)
                (_, top, left), _, (_, bot, right), _ =roi
                top, bot = sorted([top, bot])
                left, right = sorted([left, right])
                top, left, bot, right = round(top), round(left), round(bot), round(right)
                print(f'points: ({z1}, {x1}, {y1}) and ({z1}, {x2},{y2})')
                print(f'rect: ({left}, {top}) ({right}, {bot})')
                cropped = img[:, top:bot, left:right]
                x1 -= left
                x2 -= left
                y1 -= top
                y2 -= top
                try:
                    path = find_path(cropped, (z1, y1, x1), (z2, y2, x2), factor=intensity_factor, dx=x_scale, dy=y_scale, dz=z_scale)
                    if not use_fmm:
                        neck = floodfill(cropped, path, size=floodfill_radius, factor=floodfill_factor)
                    else:
                        neck = fmm_floodfill(cropped, path,fmm_ext_factor, fmm_int_factor, floodfill_radius)
                    for p in neck:
                        result[p[0]][top+p[1]][left+p[2]] = 255
                        
                    i += 1
                except:
                    print('error')
                    traceback.print_exc(file=sys.stdout)
                    return

            return Image(result, colormap='green', opacity=0.2, name=f'Result {widget_necks_segmentation.counter}')

    viewer.window.add_dock_widget(widget_necks_segmentation, name='Settings')
    
    @magicgui(call_button="Save")
    def widget_save(viewer: Viewer, binary_image_for_saving: Image) -> napari.types.LayerDataTuple:
        print('Saved')
        return (binary_image_for_saving.data, {'name': 'Saved result'})

    viewer.window.add_dock_widget(widget_save, name='Save result')

    @magicgui(call_button="Clear")
    def widget_clear_points(viewer: Viewer):
        viewer.layers['Points'].data = []
        viewer.layers['ROIs'].data = []
    
    viewer.window.add_dock_widget(widget_clear_points, name='Clear points and ROIs')
    
    return viewer
