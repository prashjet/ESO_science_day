import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.font_manager as font_manager
font_dirs = ['/Library/Fonts']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)
matplotlib.rcParams['font.family'] = 'Helvetica Neue LT Com'

import matplotlib.pyplot as plt

from PIL import Image

import pandas as pd
xl = pd.read_excel('data/doodle.xls', header=3, skip_footer=1)
xl['Name'] = xl.index
xl = xl.sort_values(['Name'])
N_ppl = len(xl)
xl = xl.set_index(np.arange(N_ppl))

def get_icon(idx_cat):
    if idx_cat==0:
        iconfile = 'data/icons/planets.png'
    elif idx_cat==1:
        iconfile = 'data/icons/stellar_evo.png'
    elif idx_cat==2:
        iconfile = 'data/icons/stellar_pops.png'
    elif idx_cat==3:
        iconfile = 'data/icons/galaxy.png'
    elif idx_cat==4:
        iconfile = 'data/icons/cosmology.png'
    elif idx_cat==5:
        iconfile = 'data/icons/spectroscopy.png'
    elif idx_cat==6:
        iconfile = 'data/icons/photo.png'
    elif idx_cat==7:
        iconfile = 'data/icons/theory.png'
    elif idx_cat==8:
        iconfile = 'data/icons/xray.png'
    elif idx_cat==9:
        iconfile = 'data/icons/optical.png'
    elif idx_cat==10:
        iconfile = 'data/icons/radio.png'
    else:
        raise ValueError('Unkown category index')
    icon = Image.open(iconfile)
    return icon

# dimensions im mm
width = 90.
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

    eso_logo = Image.open('data/icons/eso-logo-p3005.jpg')
    x0 = border_w
    w0 = eso_w
    h0 = eso_h
    y0 = 1. - border_h - h0
    ax_eso = fig.add_axes([x0, y0, w0, h0])
    ax_eso.imshow(eso_logo)
    ax_eso.axis('off')

    flags = Image.open('data/icons/eso-flags-blackframes.png')
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

    if len(xl['Name'][idx_person])>25:
        size = 16.5
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

    fig.savefig('output/badge_indiv/namebadge_{0:03d}.png'.format(idx_person))
    plt.close()

    return 0

def collate_on_page(N_ppl):

    n_per_page = 4
    n_pages = int(np.ceil(N_ppl/n_per_page))

    dx_w = 0.0000
    dx_h = 0.0000

    page_width = 2 * width + dx_w
    page_height = n_per_page * (height + dx_h)
    pagesize = (page_width/mm2inch, page_height/mm2inch)

    # transform to units of page fraction
    w = width /page_width
    h = height /page_height
    dx_w = dx_w /page_width
    dx_h = dx_h /page_height

    for i_page in range(n_pages):

        fig = plt.figure(figsize=pagesize, dpi=500)

        for i_badge in range(n_per_page):

            idx_ppl = n_per_page*i_page + i_badge
            bfile = 'output/badge_indiv/namebadge_{0:03d}.png'.format(idx_ppl)

            try:
                badge = Image.open(bfile)
            except:
                break

            y0 = dx_h/2. + i_badge * (h+dx_h)
            for x0 in [0., w+dx_w]:

                ax = fig.add_axes([x0, y0, w, h])
                ax.imshow(badge)
                ax.axis('off')

        # # add gridlines
        # ax_grid = fig.add_axes([0, 0, 1, 1])
        # ax_grid.plot([0.5, 0.5], [0, 1], '-', color='0.9', lw=3)
        # for i in range(n_per_page+1):
        #     if i==0:
        #         y0 = 0.01
        #     if i==n_per_page:
        #         y0 = 0.999
        #     else:
        #         y0 = i * (h+dx_h)
        #     ax_grid.plot([0, 1], [y0, y0], '-', color='0.9', lw=3)
        # ax_grid.set_xlim([0,1])
        # ax_grid.set_ylim([0,1])
        # ax_grid.axis('off')

        fig.savefig('output/badge_page/badges_{0:03d}.pdf'.format(i_page))
        plt.close()

    return 0

####################################################
# Main
####################################################

for idx_person in range(N_ppl):
    make_namebadge(idx_person)

collate_on_page(N_ppl)
