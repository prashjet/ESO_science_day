import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os
from IPython.core.debugger import Tracer

N_cat_pp = 3
N_per_grp = 4
N_chats_per_sesh = 8

# Utility Functions #########################################

def make_weighted_category_graph(N_ppl, cats):
    # graph to store number of pairwise mutual categories between participants
    G = nx.Graph()
    G.add_nodes_from(np.arange(N_ppl))
    for i in range(N_ppl):
        for j in range(i+1, N_ppl):
            shared_cats =  np.where(cats[i,:] * cats[j,:])
            n_shared_cats = np.size(shared_cats)
            if n_shared_cats>0:
                G.add_edge(i, j, weight=n_shared_cats)
    return G

def find_k_cliques(G, k):
    clique_iter = nx.enumerate_all_cliques(G)
    cliques = []
    for c in clique_iter:
        if len(c)<=k:
            cliques.append(c)
        else:
            break
    sizes = [len(c) for c in cliques]
    idx = np.where(np.array(sizes)==k)
    cliques = np.array(cliques)[idx]
    cliques = [tuple(c) for c in cliques]
    return tuple(cliques)

def find_n_mutual_cats(group, N_cat, cats):
    mutual = np.ones(N_cat)
    for i in group:
        mutual *= cats[i,:]
    idx = np.where(mutual)
    return len(idx[0])

def find_sum_n_pairwise_mututal(group, G):
    H = G.subgraph(group)
    return np.sum([d['weight'] for (u,v,d) in H.edges(data=True)])

def find_n_already_met(group, M):
    H = M.subgraph(group)
    return np.sum([d['weight'] for (u,v,d) in H.edges(data=True)])

def make_group_dataframe(G, M, cats, groups, N_cat, larger=False):
    N_grps = len(groups)
    n_mutual = np.zeros(N_grps)
    pw_mutual = np.zeros(N_grps)
    n_met = np.zeros(N_grps)
    for i in range(N_grps):
        n_mutual[i] = find_n_mutual_cats(groups[i], N_cat, cats)
        pw_mutual[i] = find_sum_n_pairwise_mututal(groups[i], G)
        n_met[i] = find_n_already_met(groups[i], M)
    if larger:
        df = pd.DataFrame({
            'group':list(groups),
            'n_mutual':n_mutual,
            'n_pw_mut':pw_mutual,
            'n_met':n_met,
            'cost':np.zeros(N_grps),
            'larger':np.zeros(N_grps)
        })
    else:
        df = pd.DataFrame({
            'group':list(groups),
            'n_mutual':n_mutual,
            'n_pw_mut':pw_mutual,
            'n_met':n_met,
            'cost':np.zeros(N_grps)
        })
    return df

from scipy.special import comb
max_pairs_in_grp = comb(N_per_grp, 2)
max_mutual_cats = N_cat_pp
max_sum_pw_mutual_cats = max_pairs_in_grp * N_cat_pp
def cost_function(df):
    a, b, c = 10., -3., -1.
    x = df['n_met']/max_pairs_in_grp
    y = df['n_mutual']/max_mutual_cats
    z = df['n_pw_mut']/max_sum_pw_mutual_cats
    df['cost'] = a*x + b*y + c*z
    return 0

def update_M(M, groups):
    M2 = M.copy()
    for group in groups:
        for ig, g in enumerate(group):
            for h in group[ig+1::]:
                met = M.get_edge_data(g, h)
                if met==None:
                    M2.add_edge(g, h, weight=1)
                else:
                    M2.add_edge(g, h, weight=met['weight']+1)
    return M2

def split_in_two(N_ppl):

    N_ppl_1_1 = int(48)     # i.e. 48 people in room 1, fixed
    idx_ppl_1_1 = np.random.choice(N_ppl, N_ppl_1_1, replace=False)
    idx_ppl_1_2 = np.setdiff1d(np.arange(N_ppl), idx_ppl_1_1)

    return idx_ppl_1_1, idx_ppl_1_2

# Algorithm A #########################################

def get_one_chat_grouping_A(G, M, cats, idx_grp, test_groups, N_ppl, N_cat):

    df = make_group_dataframe(G, M, cats, test_groups, N_cat)
    cost_function(df)
    df = df.sort_values(by=['cost'], ascending=True)
    df = df.reset_index(drop=True)

    groups = []
    idx_free_clq = np.where(df['n_met']>=0)[0]
    n_assigned_ppl = 0
    while (n_assigned_ppl < int(0.4*N_ppl)) and (idx_free_clq.size > 0):

        idx_clq0 = df.loc[idx_free_clq].index.tolist()[0]
        groups.append(df.loc[idx_clq0]['group'])
        n_assigned_ppl = len(np.concatenate(groups))
        busy = np.isin(
            np.column_stack(df.loc[idx_free_clq]['group']),
            df.loc[idx_clq0]['group']
        )
        busy = np.any(busy, 0)
        idx_busy = np.array(df.iloc[idx_free_clq[np.where(busy)]].index.tolist())
        idx_free_clq = np.setdiff1d(idx_free_clq, idx_busy)

    # deal with left-overs

    unassigned_ppl = np.setdiff1d(idx_grp, np.concatenate(groups))
    N_unassigned = len(unassigned_ppl)
    if N_unassigned > N_per_grp:
        N_lo_grp = int(np.floor(N_unassigned/N_per_grp))
        N_rnd = 500
        max_cost_leftover = np.inf
        for i in range(N_rnd):
            tmp = np.random.permutation(unassigned_ppl)
            lo_groups = []
            for j in range(N_lo_grp):
                tmp_grp = list(tmp[j*N_per_grp:(j+1)*N_per_grp])
                lo_groups.append(tmp_grp)
            df_tmp = make_group_dataframe(G, M, cats, lo_groups, N_cat)
            cost_function(df_tmp)
            cost_tmp = df_tmp['cost'].mean()
            if cost_tmp < max_cost_leftover:
                df_lo = df_tmp.copy()
                max_cost_leftover = cost_tmp
        for i in range(N_lo_grp):
            groups.append(tuple(df_lo.loc[i]['group']))

    unassigned_ppl = np.setdiff1d(idx_grp, np.concatenate(groups))
    N_unassigned = len(unassigned_ppl)
    if N_unassigned == N_per_grp:
            groups.append(tuple(unassigned_ppl))

    unassigned_ppl = np.setdiff1d(idx_grp, np.concatenate(groups))
    N_unassigned = len(unassigned_ppl)
    N_grp = len(groups)

    if N_unassigned > 0:
        if N_unassigned >= N_per_grp:
            print('Error: too many unasigned people...?')
        else:
            larger = np.zeros(N_grp)
            for p in unassigned_ppl:
                lo_groups = [tuple(grp)+(p,) for grp in groups]
                df_tmp = make_group_dataframe(G, M, cats, lo_groups, N_cat, larger=True)
                df_tmp['larger'] = larger
                cost_function(df_tmp)
                idx_lo = df_tmp['cost'].where(larger==0).idxmin()
                groups[idx_lo] += (p,)
                larger[idx_lo] = 1

    df = make_group_dataframe(G, M, cats, groups, N_cat)
    cost_function(df)
    return df

def get_session_grouping_A(xl, idx_grp, G, M, cats):

    N_cat =len(xl.iloc[0])-1
    N_ppl = len(idx_grp)

    cliques = find_k_cliques(G.subgraph(idx_grp), N_per_grp)
    N_cliques = len(cliques)

    group_per_chat = []
    N_test = 1000

    for i in range(N_chats_per_sesh):

        print('\t... chat', i)

        if i>0:
            M = update_M(M, group_per_chat[i-1]['group'])

        idx = np.random.choice(N_cliques, N_test, replace=False)
        test_groups = ()
        for j in idx:
            test_groups += (cliques[j],)
            idx_rnd = np.random.choice(N_ppl, N_per_grp, replace=False)
            test_groups += (tuple(idx_grp[idx_rnd]),)

        df = get_one_chat_grouping_A(
                G.subgraph(idx_grp),
                M.subgraph(idx_grp),
                cats,
                idx_grp,
                test_groups,
                N_ppl,
                N_cat
                )
        group_per_chat.append(df)

        if i==8:
            M = update_M(M, group_per_chat[i]['group'])

    return M, group_per_chat

# Algorithm B #########################################

def get_one_chat_grouping_B(G, M, cats, idx_grp, N_ppl, N_cat):
    N_test = 5000
    max_cost = np.inf
    N_grps = np.floor(len(idx_grp)/N_per_grp)
    for i in range(N_test):
        tmp = np.random.permutation(idx_grp)
        tmp_groups = np.array_split(tmp, N_grps)
        tmp_df = make_group_dataframe(G, M, cats, tmp_groups, N_cat)
        cost_function(tmp_df)
        tmp_cost = tmp_df['cost'].mean()
        if tmp_cost < max_cost:
            df = tmp_df.copy()
            max_cost = 1.*tmp_cost
    return df

def get_session_grouping_B(xl, idx_grp, G, M, cats):

    N_cat =len(xl.iloc[0])-1
    N_ppl = len(idx_grp)

    group_per_chat = []
    for i in range(N_chats_per_sesh):

        print('\t... chat', i)

        if i>0:
            M = update_M(M, group_per_chat[i-1]['group'])

        df = get_one_chat_grouping_B(
                G.subgraph(idx_grp),
                M.subgraph(idx_grp),
                cats,
                idx_grp,
                N_ppl,
                N_cat
                )
        group_per_chat.append(df)

        if i==8:
            M = update_M(M, group_per_chat[i]['group'])

    return M, group_per_chat

# Algorithm switch #########################################

def get_session_grouping(xl, idx_grp, G, M, cats, switch):
    if switch=='A':
        return get_session_grouping_A(xl, idx_grp, G, M, cats)
    if switch=='B':
        return get_session_grouping_B(xl, idx_grp, G, M, cats)

###################### process output ######################

def reorder_table_lists(df):
    for i, chat in enumerate(df):
        tab_size = [len(chat['group'][j]) for j in range(len(chat))]
        idx = np.argsort(tab_size)
        df[i] = chat.loc[idx]
        df[i] = df[i].reset_index(drop=True)
        df[i]['table'] = np.arange(len(df[i]))
    return

def add_chat_column(df):
    for i, df0 in enumerate(df):
        df0['chat'] = i
    df = pd.concat(df)
    return df

def concat_table_lists(grp_r1_s1, grp_r2_s1, grp_r1_s2, grp_r2_s2):

    grp_r1_s1 = add_chat_column(grp_r1_s1)
    grp_r2_s1 = add_chat_column(grp_r2_s1)
    grp_r1_s2 = add_chat_column(grp_r1_s2)
    grp_r2_s2 = add_chat_column(grp_r2_s2)

    grp_r1_s1['room'] = 1
    grp_r2_s1['room'] = 2
    grp_r1_s2['room'] = 1
    grp_r2_s2['room'] = 2

    grp_r1_s1['session'] = 1
    grp_r2_s1['session'] = 1
    grp_r1_s2['session'] = 2
    grp_r2_s2['session'] = 2

    df = pd.concat([grp_r1_s1, grp_r2_s1, grp_r1_s2, grp_r2_s2])
    df = df.reset_index(drop=True)

    return df

from ast import literal_eval
def print_personal_timetable(i_person, xl, plist, grp, outdir):

    if not os.path.isdir(outdir+'schedules/'):
        os.mkdir(outdir+'schedules/')

    # number of tables
    nt_r1 = len(grp[(grp['room']==1) & (grp['chat']==0) & (grp['session']==1)])
    nt_r2 = len(grp[(grp['room']==2) & (grp['chat']==0) & (grp['session']==1)])

    # get room for session 1
    if i_person in plist['r1s1']:
        room_s1 = 1
        nt_s1 = nt_r1
    else:
        room_s1 = 2
        nt_s1 = nt_r2

    # get room for session 2
    if i_person in plist['r1s2']:
        room_s2 = 1
        nt_s2 = nt_r1
    else:
        room_s2 = 2
        nt_s2 = nt_r2

    outfile = outdir+'schedules/schedule_{0:03d}.txt'.format(i_person)
    f = open(outfile, 'w')
    f.write(xl.iloc[i_person]['Name'] + '\n')
    for (sesh, room, ntables) in zip([1,2], [room_s1,room_s2], [nt_s1,nt_s2]):
        f.write('Session {i}: Room {j}\n'.format(i=sesh, j=room))
        for ic in range(N_chats_per_sesh):
            idx = ((grp['chat']==ic) & (grp['session']==sesh) & (grp['room']==room))
            tab = []
            for j in range(ntables):
                check = np.isin([i_person], literal_eval(grp[idx]['group'].iloc[j]))
                tab.append(check[0])
            tab = grp[idx]['table'][tab].iloc[0]
            f.write('\tchat {i}: table {j}\n'.format(i=ic+1, j=tab+1))
    f.close()

    return 0

def print_fancy_personal_timetable(i_person, xl, plist, grp, outdir):

    if not os.path.isdir(outdir+'fancy_schedules/'):
        os.mkdir(outdir+'fancy_schedules/')

    # number of tables
    nt_r1 = len(grp[(grp['room']==1) & (grp['chat']==0) & (grp['session']==1)])
    nt_r2 = len(grp[(grp['room']==2) & (grp['chat']==0) & (grp['session']==1)])

    # get room for session 1
    if i_person in plist['r1s1']:
        room_s1 = 1
        room_s1_str = 'Fornax'
        nt_s1 = nt_r1
    else:
        room_s1 = 2
        room_s1_str = 'Pictor/Sculptor'
        nt_s1 = nt_r2

    # get room for session 2
    if i_person in plist['r1s2']:
        room_s2 = 1
        room_s2_str = 'Fornax'
        nt_s2 = nt_r1
    else:
        room_s2 = 2
        room_s2_str = 'Pictor/Sculptor'
        nt_s2 = nt_r2

    t_s1 = np.zeros(N_chats_per_sesh)
    t_s2 = np.zeros(N_chats_per_sesh)
    arrays = zip([1,2], [room_s1,room_s2], [nt_s1,nt_s2], [t_s1, t_s2])
    for (sesh, room, ntables, t) in arrays:
        for ic in range(N_chats_per_sesh):
            idx = ((grp['chat']==ic) & (grp['session']==sesh) & (grp['room']==room))
            tab = []
            for j in range(ntables):
                check = np.isin([i_person], literal_eval(grp[idx]['group'].iloc[j]))
                tab.append(check[0])
            tab = grp[idx]['table'][tab].iloc[0]
            t[ic] = tab

    # not 0 index tables
    t_s1 +=1
    t_s2 +=1
    # add 12 to room 2
    if room_s1==2:
        t_s1 += 12
    if room_s2==2:
        t_s2 += 12

    width, height = 175., 265.
    mm2inch = 25.4

    import matplotlib
    import matplotlib.font_manager as font_manager
    font_dirs = ['/Library/Fonts']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    font_list = font_manager.createFontList(font_files)
    font_manager.fontManager.ttflist.extend(font_list)
    matplotlib.rcParams['font.family'] = 'Helvetica Neue LT Com'

    fig = plt.figure(figsize=(width/mm2inch, height/mm2inch), dpi=500)
    renderer = fig.canvas.get_renderer()

    outfile = outdir+'fancy_schedules/schedule_{0:03d}.pdf'.format(i_person)

    from PIL import Image
    eso_logo = Image.open('data/icons/eso-logo-p3005.jpg')
    x0 = 0.
    w0 = 1.7* 10.731 /width
    h0 = 1.7* 14 /height
    y0 = 1. - h0
    ax_eso = fig.add_axes([x0, y0, w0, h0])
    ax_eso.imshow(eso_logo)
    ax_eso.axis('off')

    ax_text = fig.add_axes([0, 0, 0.001, 0.001])
    ax_text.axis('off')

    # header
    event_name = ax_text.text(
            1.,
            1.,
            'ESO Garching Science Day',
            ha='right',
            va='top',
            # fontname='Helvetica',
            size=10,
            weight='bold',
            transform=fig.transFigure
    )

    bb = event_name.get_window_extent(renderer=renderer)
    x0 = bb.x0 * 1./(bb.x0+bb.width)
    y0 = bb.y0 * 1./(bb.y0+bb.height)
    y0 -= 0.01
    ax_text.text(
            x0,
            y0,
            '24 January 2018',
            ha='left',
            va='top',
            # fontname='Helvetica',
            size=10,
            weight='light',
            transform=fig.transFigure
    )

    # body
    ls = 0.025
    kw = {  'ha':'center',
            'va':'bottom',
            'size':10,
            'transform':fig.transFigure
            }
    def write_line(x0, y0, txt, weight, kw, linestep=1):
        ax_text.text(x0, y0, txt, weight=weight, **kw)
        return y0 - linestep*ls

    x0, y0 = 0.5, 0.75
    txt = 'Schedule for ' + xl.iloc[i_person]['Name']
    y0 = write_line(x0, y0, txt, 'bold', kw, 1.5)

    x0_time = 0
    x0_text = 0.1
    kw['ha']='left'

    txt = '10:30'
    y0 = write_line(x0_time, y0, txt, 'bold', kw, 0)
    txt = 'Welcome'
    y0 = write_line(x0_text, y0, txt, 'bold', kw, 1.5)

    arrays = zip([1,2],
                [room_s1_str,room_s2_str],
                [t_s1, t_s2],
                ['11:00', '14:00']
                )
    for (sesh, room, t, time) in arrays:
        y0 = write_line(x0_time, y0, time, 'bold', kw, 0)
        txt = 'Session {i} in {j}'.format(t=time, i=sesh, j=room)
        y0 = write_line(x0_text, y0, txt, 'bold', kw)
        for ic in range(N_chats_per_sesh):
            txt = '{i}. Table {j}'.format(i=ic+1, j=int(t[ic]))
            y0 = write_line(x0_text, y0, txt, 'light', kw)
        y0 -= 0.5*ls
        if sesh==1:
            txt = '12:30'
            y0 = write_line(x0_time, y0, txt, 'bold', kw, 0)
            txt = 'Photo outside Fornax or on library stairs, followed by lunch in Fornax'
            y0 = write_line(x0_text, y0, txt, 'bold', kw, 1.5)

    txt = '15:30'
    y0 = write_line(x0_time, y0, txt, 'bold', kw, 0)
    txt = 'Beer and pretzels in Fornax'
    y0 = write_line(x0_text, y0, txt, 'bold', kw)

    fig.savefig(outfile)
    plt.close()

    return 0

def plot_grouping_stats(grp, M, N_ppl, outdir):

    n_sesh = int(2 * N_chats_per_sesh)
    x = np.linspace(1, n_sesh, n_sesh)
    mx_mut = 5
    mx_pw = 30
    mx_met = 6
    n_mut = np.zeros((n_sesh, mx_mut+1))
    n_pw = np.zeros((n_sesh, mx_pw+1))
    n_met = np.zeros((n_sesh, mx_met+1))

    for i in range(8):

        idx = ((grp['chat']==i) & (grp['session']==1))
        h, e = np.histogram(grp[idx]['n_mutual'],
                range=(-0.5, mx_mut+0.5),
                bins=mx_mut+1
                )
        n_mut[i,:] = h
        h, e = np.histogram(grp[idx]['n_pw_mut'],
                range=(-0.5, mx_pw+0.5),
                bins=mx_pw+1
                )
        n_pw[i,:] = h
        h, e = np.histogram(grp[idx]['n_met'],
                range=(-0.5, mx_met+0.5),
                bins=mx_met+1
                )
        n_met[i,:] = h

        idx = ((grp['chat']==i) & (grp['session']==2))
        h, e = np.histogram(grp[idx]['n_mutual'],
                range=(-0.5, mx_mut+0.5),
                bins=mx_mut+1
                )
        n_mut[i+8,:] = h
        h, e = np.histogram(grp[idx]['n_pw_mut'],
                range=(-0.5, mx_pw+0.5),
                bins=mx_pw+1
                )
        n_pw[i+8,:] = h
        h, e = np.histogram(grp[idx]['n_met'],
                range=(-0.5, mx_met+0.5),
                bins=mx_met+1
                )
        n_met[i+8,:] = h

    d = nx.degree(M)
    allmet = np.array([d(str(i)) for i in range(N_ppl)])

    from copy import copy
    cmap = copy(plt.cm.viridis)
    cmap.set_under('w', 1.0)
    kw = {'aspect':'auto', 'vmin':1, 'vmax':20, 'cmap':cmap}

    fig, ax = plt.subplots(3, 1, figsize=(8,5), sharex=True)

    ax[0].imshow(np.flipud(n_mut.T), **kw)
    ax[1].imshow(np.flipud(n_pw.T), **kw)
    ax[2].imshow(np.flipud(n_met.T), **kw)

    ax[2].set_xlabel('Session')

    yticks = [5, 4, 3, 2, 1, 0]
    ax[0].set_yticks(yticks)
    ax[0].set_yticklabels([str(5-t) for t in yticks])
    ax[0].set_ylabel('Mutual')

    yticks = [30, 25, 20, 15, 10, 5, 0]
    ax[1].set_yticks(yticks)
    ax[1].set_yticklabels([str(30-t) for t in yticks])
    ax[1].set_ylabel('PW Mutual')

    yticks = [6, 5, 4, 3, 2, 1, 0]
    ax[2].set_yticks(yticks)
    ax[2].set_yticklabels([str(6-t) for t in yticks])
    ax[2].set_ylabel('Already Met')

    fig.tight_layout()
    fig.subplots_adjust(right=0.6)

    p0 = ax[0].get_position()
    p2 = ax[2].get_position()
    h = p0.y0 + p0.height - p2.y0
    axhist = fig.add_axes([0.67, p2.y0, 0.28, h])

    axhist.hist(allmet, range=(29.5, 48.5), bins=19, histtype='stepfilled')
    axhist.set_xlabel('Total people met')

    fig.savefig('output/grouping/plots/'+outdir+'.png')
    plt.close()

    return 0

########################################################################
# Main
########################################################################

def main(pars):

    randomseed, algorithm, outdir = pars

    # randomise
    np.random.seed(randomseed)

    # make output direc
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # read participants
    xl = pd.read_excel('data/doodle.xls', header=3, skip_footer=1)
    xl['Name'] = xl.index
    N_ppl = len(xl)
    N_cat =len(xl.iloc[0]) - 1
    xl = xl.sort_values(['Name'])
    xl = xl.set_index(np.arange(N_ppl))

    # read grouping output if already created
    if os.path.isfile(outdir+'grouping.csv'):
        groups = pd.read_csv(outdir+'grouping.csv')
        plist = np.load(outdir+'room_session_lists.npz')
        M = nx.read_edgelist(outdir+"meetings.edgelist")

    # generate grouping
    else:
        # store participant categories numpy array
        cats = np.zeros([N_ppl, N_cat])
        for i in range(N_ppl):
            idx = np.where(xl.iloc[i]=='OK')
            cats[i, idx] = 1

        # get graphs of mutual categories/meetings
        G = make_weighted_category_graph(N_ppl, cats)
        M = nx.Graph()
        M.add_nodes_from(np.arange(N_ppl))

        print('Grouping session 1...')
        idx_r1_s1, idx_r2_s1 = split_in_two(N_ppl)
        print('   ... room 1...')
        M, grp_r1_s1 = get_session_grouping(xl, idx_r1_s1, G, M, cats, algorithm)
        print('   ... room 2...')
        M, grp_r2_s1 = get_session_grouping(xl, idx_r2_s1, G, M, cats, algorithm)

        print('Grouping session 2...')
        idx_r1_s2, idx_r2_s2 = split_in_two(N_ppl)
        print('   ... room 1...')
        M, grp_r1_s2 = get_session_grouping(xl, idx_r1_s2, G, M, cats, algorithm)
        print('   ... room 2...')
        M, grp_r2_s2 = get_session_grouping(xl, idx_r2_s2, G, M, cats, algorithm)

        # reformat and store output
        for grp in (grp_r1_s1, grp_r2_s1, grp_r1_s2, grp_r2_s2):
            reorder_table_lists(grp)
        groups = concat_table_lists(grp_r1_s1, grp_r2_s1, grp_r1_s2, grp_r2_s2)
        groups.to_csv(outdir+'grouping.csv')
        np.savez(outdir+'room_session_lists.npz',
            r1s1=idx_r1_s1,
            r2s1=idx_r2_s1,
            r1s2=idx_r1_s2,
            r2s2=idx_r2_s2
        )
        plist = np.load(outdir+'room_session_lists.npz')
        nx.write_edgelist(M, outdir+"meetings.edgelist", data=True)
        M = nx.read_edgelist(outdir+"meetings.edgelist")

    # for i_person in range(N_ppl):
    #     print_personal_timetable(i_person, xl, plist, groups, outdir)
    for i_person in range(N_ppl):
        print_fancy_personal_timetable(i_person, xl, plist, groups, outdir)

    plot_grouping_stats(groups, M, N_ppl, outdir.split('/')[2])

    return 0

# try lots of random variations

# parlist = [
#     (17373, 'A', 'output/grouping/A001/'), # algo A
#     (64755, 'A', 'output/grouping/A002/'), # N_rnd = 300
#     (25734, 'A', 'output/grouping/A003/'), # n_assigned_ppl < int(0.6*N_ppl)
#     (45851, 'A', 'output/grouping/A004/'),
#     (11173, 'A', 'output/grouping/A005/'),
#     (17373, 'B', 'output/grouping/B001/'), # algo B
#     (64755, 'B', 'output/grouping/B002/'), # N_test = 2000
#     (25734, 'B', 'output/grouping/B003/'),
#     (45851, 'B', 'output/grouping/B004/'),
#     (11173, 'B', 'output/grouping/B005/'),
#     (76754, 'A', 'output/grouping/C001/'), # algo A
#     (94375, 'A', 'output/grouping/C002/'), # N_rnd = 500
#     (67585, 'A', 'output/grouping/C003/'), # n_assigned_ppl < int(0.4*N_ppl)
#     (11637, 'A', 'output/grouping/C004/'),
#     (85746, 'A', 'output/grouping/C005/'),
# ]
# parlist = [
#     (54545, 'A', 'output/grouping/D001/'), # algo A
#     (47575, 'A', 'output/grouping/D002/'), # N_rnd = 500
#     (23232, 'A', 'output/grouping/D003/'), # n_assigned_ppl < int(0.4*N_ppl)
#     (68685, 'A', 'output/grouping/D004/'),
#     (35737, 'A', 'output/grouping/D005/'),
#     (38444, 'A', 'output/grouping/D006/'),
#     (24953, 'A', 'output/grouping/D007/'),
#     (45353, 'A', 'output/grouping/D008/'),
#     (53535, 'A', 'output/grouping/D009/'),
#     (22222, 'A', 'output/grouping/D010/'),
#     (54545, 'B', 'output/grouping/E001/'), # algo B
#     (47575, 'B', 'output/grouping/E002/'), # N_test = 5000
#     (23232, 'B', 'output/grouping/E003/'),
#     (68685, 'B', 'output/grouping/E004/'),
#     (35737, 'B', 'output/grouping/E005/'),
#     (38444, 'B', 'output/grouping/E006/'),
#     (24953, 'B', 'output/grouping/E007/'),
#     (45353, 'B', 'output/grouping/E008/'),
#     (53535, 'B', 'output/grouping/E009/'),
#     (22222, 'B', 'output/grouping/E010/'),
# ]

# from multiprocessing import Pool
# pool = Pool(6)
# pool.map(main, parlist)

# print the best one
main((64755, 'A', 'output/grouping/A002/'))
