import torch
from torch import nn

from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
import math
from tqdm import tqdm

import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
class MMoE_Model(nn.Module):
    def __init__(self, input_dim=19, represent_dim=100, pair_embedding_dim=5, expert_num=5):
        super().__init__()

        self.represent_dim = represent_dim
        self.pair_embedding_dim = pair_embedding_dim
        self.input_dim = input_dim - 1
        self.expert_num = expert_num

        self.theta1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.theta2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.theta3 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.theta4 = nn.Parameter(torch.ones(1), requires_grad=True)

        self.expert_list = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.input_dim + 12 * pair_embedding_dim+20,
                          self.input_dim + 12 * pair_embedding_dim),
                nn.BatchNorm1d(self.input_dim + 12 * pair_embedding_dim),
                nn.LeakyReLU(),
                nn.Linear(self.input_dim + 12 * pair_embedding_dim, self.represent_dim),
            ) for _ in range(expert_num)
        )

        self.beta_layer = nn.Sequential(
            nn.Linear(1, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 20)
        )

        self.gate_list = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.input_dim, 20),
                nn.BatchNorm1d(20),
                nn.LeakyReLU(),
                nn.Linear(20, expert_num),
                nn.Softmax(dim=1)
            ) for _ in range(4)
        )

        self.pair_embedding_layer_list = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2, 20),
                nn.LeakyReLU(),
                nn.Linear(20, pair_embedding_dim)
            ) for _ in range(12)
        )

        self.task_layer_list = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.represent_dim, 100),
                nn.LayerNorm(100),
                nn.LeakyReLU(),
                nn.Linear(100, 50),
                nn.LayerNorm(50),
                nn.LeakyReLU(),
                nn.Linear(50, 10),
                nn.LayerNorm(10),
                nn.LeakyReLU(),
                nn.Linear(10, 1)
            ) for _ in range(4)
        )

    def forward(self, x):

        # extract the pairs from x;
        input_pair_list = []
        for i in range(6):
            input_pair_list.append(torch.stack([x[:, i], x[:, 6 + i]], dim=1))
        for i in range(6):
            input_pair_list.append(torch.stack([x[:, i], x[:, 12 + i]], dim=1))

        # convert the pairs to the embedding vectors,
        # [embeddings,x]
        input_pair_embedding_list = []

        for i, pair_embedding_layer in enumerate(self.pair_embedding_layer_list):
            input_pair_embedding_list.append(pair_embedding_layer(input_pair_list[i]))

        embeddings = torch.cat(input_pair_embedding_list, dim=1)

        beta = self.beta_layer(x[:, -1].unsqueeze(dim=-1))
        x = x[:, :-1]
        input = torch.cat([embeddings, x, beta], dim=1)
        expert_output_list = []
        for i, expert_layer in enumerate(self.expert_list):
            expert_output_list.append(expert_layer(input))
        # [expert_num,batch_size,represent_dim]
        expert_outputs = torch.stack(expert_output_list, dim=0)
        # [batch_size,expert_num,represent_dim]
        expert_outputs = expert_outputs.transpose(0, 1)

        gates = []
        for i, gate_layer in enumerate(self.gate_list):
            gates.append(gate_layer(x))
        # [4,batch_size,expert_num]
        gates = torch.stack(gates, dim=0)
        # [4,batch_size,expert_num,1]
        gates = gates.unsqueeze(3)

        # [batch_size,represent]
        task_represent_1 = torch.mean(torch.mul(gates[0], expert_outputs), dim=1)
        task_represent_2 = torch.mean(torch.mul(gates[1], expert_outputs), dim=1)
        task_represent_3 = torch.mean(torch.mul(gates[2], expert_outputs), dim=1)
        task_represent_4 = torch.mean(torch.mul(gates[3], expert_outputs), dim=1)

        output1 = self.task_layer_list[0](task_represent_1).squeeze(1)
        output2 = self.task_layer_list[1](task_represent_2).squeeze(1)
        output3 = self.task_layer_list[2](task_represent_3).squeeze(1)
        output4 = self.task_layer_list[3](task_represent_4).squeeze(1)

        # return gates
        return output1, output2, output3, output4, task_represent_1, task_represent_2, task_represent_3, task_represent_4,gates
class MMoE_Model_Expert(nn.Module):
    def __init__(self, input_dim=19, represent_dim=100, pair_embedding_dim=5, expert_num=5):
        super().__init__()

        self.represent_dim = represent_dim
        self.pair_embedding_dim = pair_embedding_dim
        self.input_dim = input_dim - 1
        self.expert_num = expert_num

        self.theta1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.theta2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.theta3 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.theta4 = nn.Parameter(torch.ones(1), requires_grad=True)

        self.expert_list = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.input_dim + 12 * pair_embedding_dim+20,
                          self.input_dim + 12 * pair_embedding_dim),
                nn.BatchNorm1d(self.input_dim + 12 * pair_embedding_dim),
                nn.LeakyReLU(),
                nn.Linear(self.input_dim + 12 * pair_embedding_dim, self.represent_dim),
            ) for _ in range(expert_num)
        )

        self.beta_layer = nn.Sequential(
            nn.Linear(1, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 20)
        )

        self.gate_list = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.input_dim, 20),
                nn.BatchNorm1d(20),
                nn.LeakyReLU(),
                nn.Linear(20, expert_num),
                nn.Softmax(dim=1)
            ) for _ in range(4)
        )

        self.pair_embedding_layer_list = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2, 20),
                nn.LeakyReLU(),
                nn.Linear(20, pair_embedding_dim)
            ) for _ in range(12)
        )

        self.task_layer_list = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.represent_dim, 100),
                nn.LayerNorm(100),
                nn.LeakyReLU(),
                nn.Linear(100, 50),
                nn.LayerNorm(50),
                nn.LeakyReLU(),
                nn.Linear(50, 10),
                nn.LayerNorm(10),
                nn.LeakyReLU(),
                nn.Linear(10, 1)
            ) for _ in range(4)
        )

    def forward(self, x):

        # extract the pairs from x;
        input_pair_list = []
        for i in range(6):
            input_pair_list.append(torch.stack([x[:, i], x[:, 6 + i]], dim=1))
        for i in range(6):
            input_pair_list.append(torch.stack([x[:, i], x[:, 12 + i]], dim=1))

        # convert the pairs to the embedding vectors,
        # [embeddings,x]
        input_pair_embedding_list = []

        for i, pair_embedding_layer in enumerate(self.pair_embedding_layer_list):
            input_pair_embedding_list.append(pair_embedding_layer(input_pair_list[i]))

        embeddings = torch.cat(input_pair_embedding_list, dim=1)

        beta = self.beta_layer(x[:, -1].unsqueeze(dim=-1))
        x = x[:, :-1]
        input = torch.cat([embeddings, x, beta], dim=1)
        expert_output_list = []
        for i, expert_layer in enumerate(self.expert_list):
            expert_output_list.append(expert_layer(input))
        # [expert_num,batch_size,represent_dim]
        expert_outputs = torch.stack(expert_output_list, dim=0)
        # [batch_size,expert_num,represent_dim]
        return expert_outputs.transpose(0, 1)

class Transfer_Model(nn.Module):
    def __init__(self, input_dim=18, represent_dim=100, pair_embedding_dim=5, expert_num=2, gate_output_dim=6):
        super().__init__()
        self.represent_dim = represent_dim
        self.pair_embedding_dim = pair_embedding_dim
        self.input_dim = input_dim - 1
        self.expert_num = expert_num

        self.theta1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.theta2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.theta3 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.theta4 = nn.Parameter(torch.ones(1), requires_grad=True)

        self.expert_list = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.input_dim + 12 * pair_embedding_dim + 20,
                          self.input_dim + 12 * pair_embedding_dim),
                nn.BatchNorm1d(self.input_dim + 12 * pair_embedding_dim),
                nn.LeakyReLU(),
                nn.Linear(self.input_dim + 12 * pair_embedding_dim, self.represent_dim),
            ) for _ in range(expert_num)
        )

        self.beta_layer = nn.Sequential(
            nn.Linear(1, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 20)
        )

        self.gate_list = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.input_dim, 20),
                nn.BatchNorm1d(20),
                nn.LeakyReLU(),
                nn.Linear(20, gate_output_dim),
                nn.Softmax(dim=1)
            ) for _ in range(4)
        )

        self.pair_embedding_layer_list = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2, 20),
                nn.LeakyReLU(),
                nn.Linear(20, pair_embedding_dim)
            ) for _ in range(12)
        )

        self.task_layer_list = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.represent_dim, 100),
                nn.LayerNorm(100),
                nn.LeakyReLU(),
                nn.Linear(100, 50),
                nn.LayerNorm(50),
                nn.LeakyReLU(),
                nn.Linear(50, 10),
                nn.LayerNorm(10),
                nn.LeakyReLU(),
                nn.Linear(10, 1)
            ) for _ in range(4)
        )

    def forward(self, x,water_expert_represents,air_expert_represents):


        # extract the pairs from x;
        input_pair_list = []
        for i in range(6):
            input_pair_list.append(torch.stack([x[:, i], x[:, 6 + i]], dim=1))
        for i in range(6):
            input_pair_list.append(torch.stack([x[:, i], x[:, 12 + i]], dim=1))


        # convert the pairs to the embedding vectors,
        # [embeddings,x]
        input_pair_embedding_list = []
        for i, pair_embedding_layer in enumerate(self.pair_embedding_layer_list):
            input_pair_embedding_list.append(pair_embedding_layer(input_pair_list[i]))
        embeddings = torch.cat(input_pair_embedding_list, dim=1)
        beta = self.beta_layer(x[:, -1].unsqueeze(dim=-1))
        x = x[:, :-1]
        input = torch.cat([embeddings, x, beta], dim=1)


        expert_output_list = []
        for i, expert_layer in enumerate(self.expert_list):
            expert_output_list .append(expert_layer(input))
        # [expert_num,batch_size,represent_dim]
        expert_outputs = torch.stack(expert_output_list , dim=0)


        # [batch_size,expert_num,represent_dim]
        expert_outputs = expert_outputs.transpose(0, 1)

        expert_outputs_1 = torch.cat([water_expert_represents[:,0,:,:],air_expert_represents[:,0,:,:],expert_outputs],dim=1)
        expert_outputs_2 = torch.cat([water_expert_represents[:,0,:,:],air_expert_represents[:,1,:,:],expert_outputs],dim=1)
        expert_outputs_3 = torch.cat([water_expert_represents[:,0,:,:],air_expert_represents[:,2,:,:],expert_outputs],dim=1)
        expert_outputs_4 = torch.cat([water_expert_represents[:,0,:,:],air_expert_represents[:,3,:,:],expert_outputs],dim=1)



        gates = []
        for i, gate_layer in enumerate(self.gate_list):
            gates.append(gate_layer(x))
        #[4,batch_size,expert_num]
        gates = torch.stack(gates, dim=0)
        #[4,batch_size,expert_num,1]
        gates = gates.unsqueeze(3)

        # [batch_size,represent]
        task_represent_1 = torch.mean(torch.mul(gates[0], expert_outputs_1), dim=1)
        task_represent_2 = torch.mean(torch.mul(gates[1], expert_outputs_2), dim=1)
        task_represent_3 = torch.mean(torch.mul(gates[2], expert_outputs_3), dim=1)
        task_represent_4 = torch.mean(torch.mul(gates[3], expert_outputs_4), dim=1)


        output1 = self.task_layer_list[0](task_represent_1).squeeze(1)
        output2 = self.task_layer_list[1](task_represent_2).squeeze(1)
        output3 = self.task_layer_list[2](task_represent_3).squeeze(1)
        output4 = self.task_layer_list[3](task_represent_4).squeeze(1)



        return output1,output2,output3,output4

