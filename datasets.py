import numpy as np
import random

import networkx as nx
import torch
from torch.utils.data import Dataset


class MyDataset_train(Dataset):
    def __init__(self, dataset, trust_dict, user_visited_dict, user_set, item_set, args):
        self.data = dataset
        self.trust_dict = trust_dict
        self.user_visited_dict = user_visited_dict
        self.item_set = item_set
        self.user_set = user_set
        self.user_list = list(user_set)
        self.item_list = list(item_set)
        self.args = args
        self.build_graph()  # 从trust_dict构建社交网络图
        self.build_mutual_friends_matrix()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item: int):
        uid = self.data[item, 0]
        iid = self.data[item, 1]

        return uid, iid

    def build_graph(self):
        """
            从trust_dict构建社交网络图
        """
        self.G = nx.Graph()
        for uid, friends in self.trust_dict.items():
            for friend in friends:
                # 确保每个用户都在图中
                if friend not in self.G:
                    self.G.add_node(friend)
                if uid not in self.G:
                    self.G.add_node(uid)
                # 添加无向边表示信任关系
                self.G.add_edge(uid, friend)

    def build_mutual_friends_matrix(self):
        """
        预构建共同好友矩阵
        """
        mutual_friends_matrix = {}
        for uid in self.G.nodes():
            mutual_friends_matrix[uid] = {}
            for friend in self.G.neighbors(uid):
                mutual_friends = len(list(nx.common_neighbors(self.G, uid, friend)))
                mutual_friends_matrix[uid][friend] = mutual_friends
        self.mutual_friends_matrix = mutual_friends_matrix

    def identify_weak_ties(self, uid, update_trust_dict, weak_threshold):
        """
        识别弱关系的用户对
        :param uid: 用户ID
        :param update_trust_dict: 用户信任字典
        :param weak_threshold: 共同好友数量阈值，低于此值视为弱关系
        :return: 弱关系的用户列表
        """
        weak_ties = set()
        if uid in update_trust_dict:
            trust_set = update_trust_dict[uid]
            for friend in trust_set:
                # 直接从预构建的矩阵中获取共同好友数量
                mutual_friends = self.mutual_friends_matrix[uid].get(friend, float('inf'))
                if mutual_friends < weak_threshold:
                    weak_ties.add(friend)
        return list(weak_ties)

    def positive_sampling_user(self, input_user_list, update_trust_dict, edge_weight):
        pos_list = []
        for i in range(len(input_user_list)):
            uid = input_user_list[i]
            if self.args.weak_threshold == 0 or uid not in update_trust_dict:
                uid_pos = uid  # 如果没有信任的用户列表，直接选择自身作为正样本
            else:
                trust_list = update_trust_dict[uid]
                # 考虑信任列表中的用户以及通过identify_weak_ties识别的弱关系用户
                candidate_list = self.identify_weak_ties(uid, update_trust_dict, self.args.weak_threshold)
                if candidate_list:  # 如果有候选用户列表且权重和不为0
                    uid_pos = np.random.choice(candidate_list)  # 根据概率选择正样本
                else:
                    uid_pos = random.sample(trust_list, 1)[0]
            pos_list.append(uid_pos)
        return pos_list

    def negative_sampling_user(self, input_user_list, update_trust_dict):
        neg_list = np.random.choice(self.user_list, len(input_user_list))
        for i in range(len(input_user_list)):
            uid = input_user_list[i]
            if uid in update_trust_dict:
                trust_set = set(update_trust_dict[uid])
                weak_set = set(self.identify_weak_ties(uid, update_trust_dict, self.args.weak_threshold))
                # 避免选择信任列表和弱关系列表中的用户作为负样本
                if neg_list[i] in trust_set or neg_list[i] in weak_set:
                    uid_neg = random.sample(self.user_set - trust_set - weak_set, 1)[0]
                    while uid_neg in trust_set or uid_neg in weak_set:
                        uid_neg = random.sample(self.user_set - trust_set - weak_set, 1)[0]
                    neg_list[i] = uid_neg
        return neg_list

    def negative_sampling_item(self, input_user_list):
        neg_list = np.random.choice(self.item_list, len(input_user_list))
        output_sampling_result = []
        for i in range(len(input_user_list)):
            visited_set = self.user_visited_dict[input_user_list[i]]
            if neg_list[i] in visited_set:
                iid_neg = random.sample(self.item_set, 1)[0]
                while iid_neg in visited_set:
                    iid_neg = random.sample(self.item_set, 1)[0]
                output_sampling_result.append(iid_neg)
            else:
                output_sampling_result.append(neg_list[i])
        return output_sampling_result


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item: int):
        uid = self.data[item, 0]
        iid = self.data[item, 1]
        return [uid, iid]


def get_collator_train():
    def collator(data_points):
        packed_feat = []
        packed_trust_list = []
        for feat, trust_list in data_points:
            packed_feat.append(torch.LongTensor(np.reshape(np.array(feat), [1, -1])))
            packed_trust_list.append(trust_list)
        packed_feat = torch.cat(packed_feat, dim=0)
        return packed_feat, packed_trust_list

    return collator


def get_collator():
    def collator(data_points):
        packed_feat = []
        for feat in data_points:
            packed_feat.append(torch.LongTensor(np.reshape(np.array(feat), [1, -1])))
        packed_feat = torch.cat(packed_feat, dim=0)
        return packed_feat

    return collator
