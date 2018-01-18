import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from PIL import Image

import pandas as pd
xl = pd.read_excel('doodle.xls', header=3, skip_footer=1)
xl['Name'] = xl.index
N_ppl = len(xl)
xl = xl.set_index(np.arange(N_ppl))

outdir = 'badges/'

def get_icon(idx_cat):
    if idx_cat==0:
        iconfile = 'icons/planets.png'
    elif idx_cat==1:
        iconfile = 'icons/stellar_evo.png'
    elif idx_cat==2:
        iconfile = 'icons/stellar_pops.png'
    elif idx_cat==3:
        iconfile = 'icons/galaxy.png'
    elif idx_cat==4:
        iconfile = 'icons/cosmology.png'
    elif idx_cat==5:
        iconfile = 'icons/spectroscopy.png'
    elif idx_cat==6:
        iconfile = 'icons/photo.png'
    elif idx_cat==7:
        iconfile = 'icons/theory.png'
    elif idx_cat==8:
        iconfile = 'icons/xray.png'
    elif idx_cat==9:
        iconfile = 'icons/optical.png'
    elif idx_cat==10:
        iconfile = 'icons/radio.png'
    else:
        raise ValueError('Unkown category index')
    icon = Image.open(iconfile)
    return icon

# dimensions im mm
width = 89.75
height = 60.

# dimensions in frac of width/height
border_w = 4.763 /width
border_h = 4.763 /height
eso_w = 10.731 /width
eso_h = 14 /height
flag_w = 70. /width
flag_h = 2. /height
icon_w = 10. /width
icon_h = 14. /height
icon_dw = 4. /width
mm2inch = 25.4

def make_namebadge(idx_person):

    fig = plt.figure(figsize=(width/mm2inch, height/mm2inch), dpi=500)
    renderer = fig.canvas.get_renderer()

    eso_logo = Image.open('icons/eso-logo-p3005.jpg')
    x0 = border_w
    w0 = eso_w
    h0 = eso_h
    y0 = 1. - border_h - h0
    ax_eso = fig.add_axes([x0, y0, w0, h0])
    ax_eso.imshow(eso_logo)
    ax_eso.axis('off')

    flags = Image.open('icons/eso-flags-blackframes.png')
    w0 = flag_w
    x0 = 1. - flag_w - border_w
    y0 = border_h
    h0 = flag_h
    ax_flag = fig.add_axes([x0, y0, w0, h0])
    ax_flag.imshow(flags)
    ax_flag.axis('off')

    ax_text = fig.add_axes([0, 0, 0.001, 0.001])
    ax_text.axis('off')
    event_name = ax_text.text(
            1.-border_w,
            1.-border_h,
            'ESO Garching Science Day',
            ha='right',
            va='top',
            # fontname='Helvetica',
            size=8,
            weight='bold',
            transform=fig.transFigure
    )

    bb = event_name.get_window_extent(renderer=renderer)

    x0 = bb.x0 * (1.-border_w)/(bb.x0+bb.width)
    y0 = bb.y0 * (1.-border_h)/(bb.y0+bb.height)
    y0 -= 0.015
    ax_text.text(
            x0,
            y0,
            '24 January 2018',
            ha='left',
            va='top',
            # fontname='Helvetica',
            size=8,
            weight='light',
            transform=fig.transFigure
    )

    if xl['Name'][idx_person]=='Maria Giulia Ubeira Gabellini':
        size = 17.
    else:
        size = 18.5

    ax_text.text(
            0.5,
            0.525,
            xl['Name'][idx_person],
            ha='center',
            va='center',
            # fontname='Helvetica',
            size=size,
            weight='medium',
            transform=fig.transFigure
    )

    idx_cats = np.where(xl.loc[idx_person]=='OK')
    idx_cats = idx_cats[0]
    n_cats = len(idx_cats)
    n_cats % 2
    if n_cats % 2:
        x0 = 0.5 - np.floor(n_cats/2.)*(icon_w+icon_dw) - 0.5*icon_w
    else:
        x0 = 0.5 - (n_cats/2.)*(icon_w+icon_dw) + 0.5*icon_dw
    y0 = 11. / height
    for idx in idx_cats:
        ax_icon = fig.add_axes([x0, y0, icon_w, icon_h])
        icon = get_icon(idx)
        ax_icon.imshow(icon)
        ax_icon.axis('off')
        x0 += (icon_w+icon_dw)

    fig.savefig(outdir+'namebadge_{i}.png'.format(i=idx_person))
    plt.close()

    return 0

####################################################
# Main
####################################################

for idx_person in range(N_ppl):
    make_namebadge(idx_person)


####################################################
