import copy
import logging
import random
import numpy as np


def gen_pc_list(n, m, ws, b_mean, b_sig, d_max, d_uniform=True):
    users_list = [i for i in range(n)]
    _pc_list = []
    for i in range(m):
        flag = 0
        while flag == 0:
            flag = 1
            sender, receiver = random.sample(users_list, 2)
            for __pc in _pc_list:
                if sender == __pc[0] and receiver == __pc[1]:
                    flag = 0
                    break
                if receiver == __pc[0] and sender == __pc[1]:
                    flag = 0
                    break
        w = (ws[1] - ws[0]) * random.random() + ws[0]

        b_ji = b_mean + b_sig * np.random.randn(1)
        while b_ji < 0:
            b_ji = b_mean + b_sig * np.random.randn(1)

        if d_uniform:
            d_ij = random.randint(1, d_max)
        else:
            d_ij = d_max
        _pc = [sender, receiver, w, b_ji[0], d_ij]
        _pc_list.append(_pc[:])

    return _pc_list


def logging_path(nodes_or_pcs, mode='other'):
    if mode == 'nodes_ring':
        output = f'circuit found【{nodes_or_pcs[0]}'
        for node in nodes_or_pcs[1:]:
            output += f'——>{node}'
        output += '】'
        return

    if isinstance(nodes_or_pcs[0], int):
        output = f'{nodes_or_pcs[0]}'
        for node in nodes_or_pcs[1:]:
            output += f'——>{node}'

    else:
        output = f'circuit found【{nodes_or_pcs[0][0]}——>{nodes_or_pcs[0][1]}'
        for pc in nodes_or_pcs[1:]:
            output += f'——>{pc[1]}'
        output += '】'


def alg_exact(_pc_list):
    n_max = max([max(_pc[0], _pc[1]) for _pc in _pc_list])
    n_min = min([min(_pc[0], _pc[1]) for _pc in _pc_list])
    n = n_max - n_min + 1

    def gen_all_list(max_limited):
        all_lists = []

        def gen_list(length, index, curr_list):

            if index == length:
                all_lists.append(curr_list[:])
                return

            for i in range(max_limited[index] + 1):
                curr_list[index] = i
                gen_list(length, index + 1, curr_list)

        gen_list(len(max_limited), 0, [0] * len(max_limited))

        return all_lists

    def calc_utility(_pc_list, curr_d_list):
        node_out = [0 for _ in range(n)]
        node_in = [0 for _ in range(n)]
        for index, amount in enumerate(curr_d_list):
            node_out[_pc_list[index][0] - n_min] += amount
            node_in[_pc_list[index][1] - n_min] += amount
        if node_in == node_out and sum(node_in) != 0:
            utility = sum([_pc_list[i][2] * curr_d_list[i] / (_pc_list[i][3] + 1) for i in range(len(_pc_list))])

        else:
            utility = 0
        return utility

    all_list = gen_all_list([_pc[4] for _pc in _pc_list])

    max_utility = 0
    for curr_d_list in all_list:
        utility = calc_utility(_pc_list, curr_d_list)
        if utility > max_utility:
            max_utility = utility

    return max_utility


def alg_baseline_random(_pc_table):
    _pc_list = table_to_pc_list(_pc_table)

    def find_closed(_pc_list, t):
        visited_node = []
        visited_pc = []

        _pc = random.choice(_pc_list)

        visited_pc.append(_pc[:])
        source = _pc[0]
        dest = _pc[1]

        while dest != source:
            visited_node.append(dest)
            logging_path(visited_node)
            next_pc_list = [_pc for _pc in _pc_list if _pc[0] == dest and _pc[1] not in visited_node]
            if len(next_pc_list):
                next_pc = random.choice(next_pc_list)
                visited_pc.append(next_pc[:])
                dest = next_pc[1]
            else:
                temp = _pc_list[:]
                temp.remove(random.choice(visited_pc))

                return 0, temp

        logging_path(visited_pc)

        min_d = min([_pc[4] for _pc in visited_pc])
        utility = sum([vis_pc[2] * min_d / (vis_pc[3] + 1) for vis_pc in visited_pc])

        _pc_list = [_pc for _pc in _pc_list if _pc not in visited_pc]
        for vis_pc in visited_pc:
            temp = vis_pc[:]
            temp[4] -= min_d
            print("temp[4]=",temp[4])
            if not isinstance(temp[4], int):
                breakpoint()
            t.append([temp[0], temp[1], min_d])
            if temp[4] != 0:
                _pc_list.append(temp)
        return utility, _pc_list

    total_utility = 0
    new_pc_list = _pc_list[:]
    trans = []

    while len(new_pc_list):
        utility, new_pc_list = find_closed(new_pc_list[:], trans)
        total_utility += utility

    return total_utility, trans


def alg_baseline_greedy(_pc_table):
    _pc_list = table_to_pc_list(_pc_table)

    def find_closed(_pc_list, t):
        visited_node = []
        visited_pc = []

        max_d_pc = max(_pc_list, key=lambda x: x[4])

        visited_node.append(max_d_pc[0])
        visited_pc.append(max_d_pc[:])
        dest = max_d_pc[1]

        while dest not in visited_node:
            visited_node.append(dest)
            logging_path(visited_node)
            next_pc_list = [_pc for _pc in _pc_list if _pc[0] == dest]
            if len(next_pc_list):
                max_d_next_pc = max(next_pc_list, key=lambda x: x[4])
                visited_pc.append(max_d_next_pc[:])
                dest = max_d_next_pc[1]
            else:
                min_d_pc = min(visited_pc, key=lambda x: x[4])
                temp = _pc_list[:]
                temp.remove(min_d_pc)
                return 0, temp

        while len(visited_pc) and visited_pc[0][0] != dest:
            visited_pc.pop(0)
        logging_path(visited_pc)
        min_d = min([_pc[4] for _pc in visited_pc])
        utility = sum([vis_pc[2] * min_d / (vis_pc[3] + 1) for vis_pc in visited_pc])

        _pc_list = [_pc for _pc in _pc_list if _pc not in visited_pc]
        for vis_pc in visited_pc:
            temp = vis_pc[:]
            temp[4] -= min_d
            if not isinstance(temp[4], int):
                breakpoint()
            t.append([temp[0], temp[1], min_d])
            if temp[4] != 0:
                _pc_list.append(temp)

        return utility, _pc_list

    total_utility = 0
    new_pc_list = _pc_list[:]
    trans = []
    while len(new_pc_list):
        utility, new_pc_list = find_closed(new_pc_list[:], trans)
        total_utility += utility

    return total_utility, trans


def alg_circuit_greedy_table(_pc_table):
    def dfs(_source, cur_node, pre_visited, visited_nodes):
        nonlocal cg_circuit
        if len(cg_circuit):
            return False
        if access_table[cur_node] == -1:
            return False
        if cur_node == _source:
            pre_visited.append(cur_node)
            cg_circuit = pre_visited[:]
            return True
        pre_visited.append(cur_node)
        visited_nodes.append(cur_node)
        if cur_node in _pc_table:
            for node in _pc_table[cur_node]:
                if access_table[node] != -1 and node not in visited_nodes:
                    dfs(_source, node, pre_visited[:], visited_nodes[:])

        if access_table[cur_node] != 1:
            access_table[cur_node] = -1

    total_utility = 0
    RP = {}
    trans = []
    while len(_pc_table):
        max_d = 0
        for _start in _pc_table:
            for _end in _pc_table[_start]:
                if _pc_table[_start][_end][2] > max_d:
                    max_d = _pc_table[_start][_end][2]
                    max_d_start = _start
                    max_d_end = _end

        cg_circuit = []
        access_table = {}
        for _start in _pc_table:
            for _end in _pc_table[_start]:
                if _start not in access_table:
                    access_table[_start] = 0
                if _end not in access_table:
                    access_table[_end] = 0
        dfs(max_d_start, max_d_end, [max_d_start], [])

        if len(cg_circuit):
            min_d = min([_pc_table[cg_circuit[i]][cg_circuit[i + 1]][2] for i in range(len(cg_circuit) - 1)])
            utility = sum([_pc_table[cg_circuit[i]][cg_circuit[i + 1]][0] * min_d / (_pc_table[cg_circuit[i]][cg_circuit[i + 1]][1] + 1) for i in range(len(cg_circuit) - 1)])
            total_utility += utility
            for i in range(len(cg_circuit) - 1):
                _pc_table[cg_circuit[i]][cg_circuit[i + 1]][2] -= min_d
                trans.append([cg_circuit[i], cg_circuit[i + 1], min_d])
                if _pc_table[cg_circuit[i]][cg_circuit[i + 1]][2] == 0:
                    _pc_table[cg_circuit[i]].pop(cg_circuit[i + 1])
                    if len(_pc_table[cg_circuit[i]]) == 0:
                        _pc_table.pop(cg_circuit[i])
        else:
            if max_d_start not in RP:
                RP[max_d_start] = {max_d_end: _pc_table[max_d_start][max_d_end][:]}
            else:
                RP[max_d_start][max_d_end] = _pc_table[max_d_start][max_d_end][:]

            _pc_table[max_d_start].pop(max_d_end)
            if len(_pc_table[max_d_start]) == 0:
                _pc_table.pop(max_d_start)

    return RP, total_utility, trans


def alg_circuit_greedy_table_new(_pc_table, ret_rp=0):
    def dfs_find_one(_source, cur_node, pre_visited, visited_nodes):
        nonlocal cg_circuit, access_table
        if len(cg_circuit):
            return False
        if access_table[cur_node] == -1:
            return False
        if cur_node == _source:
            pre_visited.append(cur_node)
            cg_circuit = pre_visited[:]
            return True

        pre_visited.append(cur_node)
        visited_nodes.append(cur_node)
        if cur_node in _pc_table:
            for node in _pc_table[cur_node]:
                if access_table[node] != -1 and node not in visited_nodes:
                    dfs_find_one(_source, node, pre_visited[:], visited_nodes[:])

        if access_table[cur_node] != 1:
            access_table[cur_node] = -1

    total_utility = 0
    RP = {}
    trans = []
    while len(_pc_table):
        all_circuits = []
        for start in list(_pc_table):
            for end in list(_pc_table[start]):
                cg_circuit = []
                access_table = {}
                for _start in _pc_table:
                    for _end in _pc_table[_start]:
                        if _start not in access_table:
                            access_table[_start] = 0
                        if _end not in access_table:
                            access_table[_end] = 0
                dfs_find_one(start, end, [start], [])
                if len(cg_circuit):
                    all_circuits.append(cg_circuit[:])
                else:
                    if start not in RP:
                        RP[start] = {end: _pc_table[start][end][:]}
                    else:
                        RP[start][end] = _pc_table[start][end][:]
                    _pc_table[start].pop(end)
                    if len(_pc_table[start]) == 0:
                        _pc_table.pop(start)

        if len(all_circuits) == 0:
            if len(_pc_table) != 0:
                breakpoint()
            break
        else:
            max_utility = 0
            for circuit in all_circuits:
                min_d = min([_pc_table[circuit[i]][circuit[i + 1]][2] for i in range(len(circuit) - 1)])
                utility = sum([_pc_table[circuit[i]][circuit[i + 1]][0] * min_d / (_pc_table[circuit[i]][circuit[i + 1]][1] + 1) for i in range(len(circuit) - 1)])
                if utility > max_utility:
                    max_utility = utility
                    max_circuit = circuit[:]
                    sub_min_d = min_d

            total_utility += max_utility
            for i in range(len(max_circuit) - 1):
                _pc_table[max_circuit[i]][max_circuit[i + 1]][2] -= sub_min_d
                trans.append([max_circuit[i], max_circuit[i + 1], sub_min_d])
                if _pc_table[max_circuit[i]][max_circuit[i + 1]][2] == 0:
                    _pc_table[max_circuit[i]].pop(max_circuit[i + 1])
                    if len(_pc_table[max_circuit[i]]) == 0:
                        _pc_table.pop(max_circuit[i])

    if not ret_rp:
        return total_utility, trans
    return total_utility, trans


def pc_list_to_table(_pc_list):
    _pc_table = {}
    for _pc in _pc_list:
        if _pc[0] not in _pc_table:
            _pc_table[_pc[0]] = {_pc[1]: [_pc[2], _pc[3], _pc[4]]}
        else:
            _pc_table[_pc[0]][_pc[1]] = [_pc[2], _pc[3], _pc[4]]
    return _pc_table


def table_to_pc_list(_pc_table):
    _pc_list = []
    for start in _pc_table:
        for end in _pc_table[start]:
            _pc_list.append([start, end, *_pc_table[start][end]])
    return _pc_list


def alg_all_greedy_table(_pc_table):
    total_utility = 0

    def find_pc_circuit(start, end):
        pc_circuits = []

        def dfs(_source, cur_node, pre_visited, visited_nodes):
            if access_table[cur_node] == -1:
                return False
            if cur_node == _source:
                pre_visited.append(cur_node)
                for pre in pre_visited:
                    access_table[pre] = 1
                for a_cir in all_circuits:
                    a = a_cir[1:]
                    b = pre_visited[1:]
                    if set(a) == set(b):
                        index_t = b.index(a[0])
                        new_b = b[index_t:] + b[:index_t]
                        if a == new_b:
                            return True

                pc_circuits.append(pre_visited[:])
                return True

            pre_visited.append(cur_node)
            visited_nodes.append(cur_node)
            if cur_node in _pc_table:
                for node in _pc_table[cur_node]:
                    if access_table[node] != -1 and node not in visited_nodes:
                        dfs(_source, node, pre_visited[:], visited_nodes[:])

            if access_table[cur_node] != 1:
                access_table[cur_node] = -1

        access_table = {}
        for _start in _pc_table:
            for _end in _pc_table[_start]:
                if _start not in access_table:
                    access_table[_start] = 0
                if _end not in access_table:
                    access_table[_end] = 0
        dfs(start, end, [start], [])
        return pc_circuits

    all_circuits = []
    trans = []
    for start in _pc_table:
        for end in _pc_table[start]:
            pc_circuits = find_pc_circuit(start, end)
            all_circuits += pc_circuits

    while len(all_circuits):
        max_utiltiy = 0
        max_circuit = []
        circuit_remove = []
        for circuit in all_circuits:
            remove_flag = 0
            for i in range(len(circuit) - 1):
                if circuit[i] not in _pc_table:
                    remove_flag = 1
                elif circuit[i + 1] not in _pc_table[circuit[i]]:
                    remove_flag = 1
                elif _pc_table[circuit[i]][circuit[i + 1]][2] == 0:
                    remove_flag = 1
            if remove_flag:
                circuit_remove.append(circuit)
                continue
            min_d = min([_pc_table[circuit[i]][circuit[i + 1]][2] for i in range(len(circuit) - 1)])
            utility = sum([_pc_table[circuit[i]][circuit[i + 1]][0] * min_d / (_pc_table[circuit[i]][circuit[i + 1]][1] + 1) for i in range(len(circuit) - 1)])
            if utility > max_utiltiy:
                max_utiltiy = utility
                max_circuit = circuit[:]
                sub_min_d = min_d

        if len(max_circuit):
            total_utility += max_utiltiy
            circuit_remove.append(max_circuit)

        for i in range(len(max_circuit) - 1):
            _pc_table[max_circuit[i]][max_circuit[i + 1]][2] -= sub_min_d
            trans.append([max_circuit[i], max_circuit[i + 1], sub_min_d])
            if _pc_table[max_circuit[i]][max_circuit[i + 1]][2] == 0:
                _pc_table[max_circuit[i]].pop(max_circuit[i + 1])
                if not len(_pc_table[max_circuit[i]]):
                    _pc_table.pop(max_circuit[i])

        for cir in circuit_remove:
            all_circuits.remove(cir)

    RP = copy.deepcopy(_pc_table)
    return RP, total_utility, trans


def alg_revive_exact_new_pc(_pc_table, x=7, y=7):
    _pc_list = table_to_pc_list(_pc_table)
    total_utility = 0
    while len(_pc_list):
        if len(_pc_list) > x:
            pc_xs = random.sample(_pc_list, x)
        else:
            pc_xs = _pc_list[:]
        for pc in pc_xs:
            _pc_list.remove(pc)

        while len(pc_xs):
            pc_x_ys = []
            remove_pc_list = []
            for pc in pc_xs:
                if pc[4] > y:
                    new_pc = pc[:]
                    new_pc[4] = y
                else:
                    new_pc = pc[:]
                pc[4] = pc[4] - new_pc[4]
                pc_x_ys.append(new_pc)
                if pc[4] == 0:
                    remove_pc_list.append(pc)
            for _pc in remove_pc_list:
                pc_xs.remove(_pc)

            total_utility += alg_exact(copy.deepcopy(pc_x_ys))

    return total_utility


def alg_DC(pc_table, L=12):
    def calc_sd(G, p, G1_p):
        total_d = 0
        for start in G:
            for end in G[start]:
                if start == p and end not in G1_p:
                    total_d += G[start][end][2]
                if end == p and start not in G1_p:
                    total_d += G[start][end][2]
        return total_d

    def get_min_p(G, G1_p):
        all_ps = []
        for start in G:
            for end in G[start]:
                if start not in all_ps and start not in G1_p:
                    all_ps.append(start)
                if end not in all_ps and end not in G1_p:
                    all_ps.append(end)

        min_sd = float('inf')
        for p in all_ps:
            sd = calc_sd(G, p, G1_p)
            if sd < min_sd:
                min_sd = sd
                min_p = p
        return min_p

    def split(pc_table):
        all_ps = []
        for start in pc_table:
            for end in pc_table[start]:
                if start not in all_ps:
                    all_ps.append(start)
                if end not in all_ps:
                    all_ps.append(end)

        if len(all_ps) > L:
            G0 = {}
            G1 = {}
            G1_p = []
            RG = {}

            while len(G1_p) < len(all_ps) - 1:
                p = get_min_p(pc_table, G1_p)
                G1_p.append(p)
                all_ps.remove(p)

            for start in list(pc_table):
                for end in list(pc_table[start]):
                    if start in G1_p:
                        if end in G1_p:
                            if start not in G1:
                                G1[start] = {}
                            G1[start][end] = pc_table[start][end]
                        else:
                            if start not in RG:
                                RG[start] = {}
                            RG[start][end] = pc_table[start][end]
                    else:
                        if end in G1_p:
                            if start not in RG:
                                RG[start] = {}
                            RG[start][end] = pc_table[start][end]
                        else:
                            if start not in G0:
                                G0[start] = {}
                            G0[start][end] = pc_table[start][end]

            RP1, util1, trans1 = split(copy.deepcopy(G0))
            RP2, util2, trans2 = split(copy.deepcopy(G1))
            new_RG = {}
            for G in [RG, RP1, RP2]:
                for start in G:
                    for end in G[start]:
                        if start not in new_RG:
                            new_RG[start] = {}
                        new_RG[start][end] = copy.deepcopy(G[start][end])

            RP3, util3, trans3 = alg_circuit_greedy_table(copy.deepcopy(new_RG))
            all_util = util1 + util2 + util3
            return RP3, all_util, trans1 + trans2 + trans3
        else:

            RP, util, trans = alg_all_greedy_table(copy.deepcopy(pc_table))
            return RP, util, trans

    RP, util, trans = split(copy.deepcopy(pc_table))
    return util, trans


def write_log(file_name, current_time, sentence):
    with open(f'Results/{file_name}.txt', 'a') as f:
        f.writelines(sentence)

    with open(f'Results/Backup/{current_time}/{file_name}.txt', 'a') as f:
        f.writelines(sentence)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='【%(asctime)s】【%(levelname)s】【"%(pathname)s:%(lineno)d"】 ： %(message)s', datefmt="%H:%M:%S")
    pass
