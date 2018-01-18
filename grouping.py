import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer

N_cat_pp = 3
N_per_grp = 4
N_chats_per_sesh = 8

def make_weighted_category_graph(N_ppl, cats):
    # graph to store # of pairwise mutual categories between participants
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

def get_one_chat_grouping(G, M, cats, idx_grp, test_groups, N_ppl, N_cat):

    df = make_group_dataframe(G, M, cats, test_groups, N_cat)
    cost_function(df)
    df = df.sort_values(by=['cost'], ascending=True)
    df = df.reset_index(drop=True)

    groups = []
    idx_free_clq = np.where(df['n_met']>=0)[0]
    n_assigned_ppl = 0
    while (n_assigned_ppl < int(0.6*N_ppl)) and (idx_free_clq.size > 0):

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
        N_rnd = 300
        np.random.seed(30121988)
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

def split_in_two(N_ppl, seed):

    np.random.seed(seed)
    N_ppl_1_1 = int(48)
    idx_ppl_1_1 = np.random.choice(N_ppl, N_ppl_1_1, replace=False)
    idx_ppl_1_2 = np.setdiff1d(np.arange(N_ppl), idx_ppl_1_1)

    return idx_ppl_1_1, idx_ppl_1_2

def get_session_grouping(xl, idx_grp, G, M, cats):

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

        df = get_one_chat_grouping(
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

def reorder_table_lists(df):
    for i, chat in enumerate(df):
        tab_size = [len(chat['group'][j]) for j in range(len(chat))]
        idx = np.argsort(tab_size)
        df[i] = chat.loc[idx]
        df[i] = df[i].reset_index(drop=True)
    return 0

# To complete
def print_personal_timetable(i_person, xl, g1_s1, g2_s1, g1_s2, g2_s2):

    idx_r1_s1, grp_r1_s1 = g1_s1
    idx_r2_s1, grp_r2_s1 = g2_s1
    idx_r1_s2, grp_r1_s2 = g1_s2
    idx_r2_s2, grp_r2_s2 = g2_s2

    if i_person in idx_r1_s1:
        idx_s1, grp_s1 = idx_r1_s1, grp_r1_s1
        room_s1 = 'Room 1'
    else:
        idx_s1, grp_s1 = idx_r2_s1, grp_r2_s1
        room_s1 = 'Room 2'
    nt_s1 = len(grp_s1[0])
    if i_person in idx_r1_s2:
        idx_s2, grp_s2 = idx_r1_s2, grp_r1_s2
        room_s2 = 'Room 1'
    else:
        idx_s2, grp_s2 = idx_r2_s2, grp_r2_s2
        room_s2 = 'Room 2'
    nt_s2 = len(grp_s2[0])

    print(xl.iloc[i_person]['Name'])
    print('Session 1:', room_s1)
    for ic in range(N_chats_per_sesh):
        tab = [np.isin(i_person, grp_s1[ic]['group'][j]) for j in range(nt_s1)]
        tab = np.where(tab)
        tab = tab[0][0]
        print('Table', tab)
    print('Session 2:', room_s2)
    for ic in range(N_chats_per_sesh):
        tab = [np.isin(i_person, grp_s2[ic]['group'][j]) for j in range(nt_s2)]
        tab = np.where(tab)
        tab = tab[0][0]
        print('Table', tab)

    return 0

# TO-DO...
def plot_grouping_stats(g1_s1, g2_s1, g1_s2, g2_s2):

    idx_r1_s1, grp_r1_s1 = g1_s1
    idx_r2_s1, grp_r2_s1 = g2_s1
    idx_r1_s2, grp_r1_s2 = g1_s2
    idx_r2_s2, grp_r2_s2 = g2_s2

    fig, ax = plt.subplots(3, 1, sharex=True)

    x1 = np.linspace(1,8,8)
    x2 = np.linspace(10, 17, 8)

    y_mean = [np.mean(grp_r1_s1[i]['n_met']) for i in range(N_chats_per_sesh)]
    y_min = [np.min(grp_r1_s1[i]['n_met']) for i in range(N_chats_per_sesh)]
    y_min = [np.min(grp_r1_s1[i]['n_met']) for i in range(N_chats_per_sesh)]


########################################################################
# Main
########################################################################

xl = pd.read_excel('doodle.xls', header=3, skip_footer=1)
xl.loc['Ralf Siebenmorgen']['Spectroscopy'] = 'OK'
xl.loc['Ralf Siebenmorgen']['Stellar Structure & Evolution'] = 'OK'
xl['Name'] = xl.index
N_ppl = len(xl)
N_cat =len(xl.iloc[0]) - 1
xl = xl.set_index(np.arange(N_ppl))

cats = np.zeros([N_ppl, N_cat])
for i in range(N_ppl):
    idx = np.where(xl.iloc[i]=='OK')
    cats[i, idx] = 1

G = make_weighted_category_graph(N_ppl, cats)

M = nx.Graph()
M.add_nodes_from(np.arange(N_ppl))

# TO-DO: remove certain people from morning/afternoon session as requested

print('Grouping session 1...')
random_seed = 342344
idx_r1_s1, idx_r2_s1 = split_in_two(N_ppl, random_seed)
print('   ... room 1...')
M, grp_r1_s1 = get_session_grouping(xl, idx_r1_s1, G, M, cats)
print('   ... room 2...')
M, grp_r2_s1 = get_session_grouping(xl, idx_r2_s1, G, M, cats)

print('Grouping session 2...')
random_seed = 576785
idx_r1_s2, idx_r2_s2 = split_in_two(N_ppl, random_seed)
print('   ... room 1...')
M, grp_r1_s2 = get_session_grouping(xl, idx_r1_s2, G, M, cats)
print('   ... room 2...')
M, grp_r2_s2 = get_session_grouping(xl, idx_r2_s2, G, M, cats)

for grp in (grp_r1_s1, grp_r2_s1, grp_r1_s2, grp_r2_s2):
    reorder_table_lists(grp)

g1_s1 = (idx_r1_s1, grp_r1_s1)
g2_s1 = (idx_r2_s1, grp_r2_s1)
g1_s2 = (idx_r1_s2, grp_r1_s2)
g2_s2 = (idx_r2_s2, grp_r2_s2)

i_person = 34
print_personal_timetable(i_person, xl, g1_s1, g2_s1, g1_s2, g2_s2)
