import argparse
import math
import os

import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from torch import nn
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datasets import get_train_test_data2, MMoE_Dataset,get_train_test_data1
from models import MMoE_Model, Transfer_Model, MMoE_Model_Expert
file_path = "./data/mutilsolvent2.csv"
data = pd.read_csv(file_path).values
from sklearn.model_selection import train_test_split


def get_train_test_data(file_path, test_size,seed,dataset_num):
    data = pd.read_csv(file_path).values
    x_train_list=[]
    y_train_list=[]
    x_test_list=[]
    y_test_list=[]
    for i in range(15):
        if i == dataset_num:
            cur_data = data[35*i:35*(i+1),:]
            transfer_x,transfer_y = cur_data[:, :-4], cur_data[:, -4:]
        else:
            cur_data = data[35*i:35*(i+1),:]
            X, y = cur_data[:, :-4], cur_data[:, -4:]
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=seed)
            x_train_list.append(x_train)
            y_train_list.append(y_train)
            x_test_list.append(x_test)
            y_test_list.append(y_test)
    x_train, x_test, y_train, y_test = np.concatenate(x_train_list),np.concatenate(x_test_list),np.concatenate(y_train_list),np.concatenate(y_test_list)
    print(f"dataset num {dataset_num}")
    print(f"the number of train samples is: {len(x_train)}")
    print(f"the number of test samples is: {len(x_test)}")
    print(f"the number of transfer samples is: {len(x_test)}")
    return x_train, x_test, y_train, y_test,transfer_x,transfer_y
for i in range(15):
    x_train, x_test, y_train, y_test,transfer_X,transfer_y = get_train_test_data(file_path,0.2,42,i)


def create_dir(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))


def trainer(water_model, water_select_expert, air_model, air_select_expert, train_loader, model, model_save_path,
            device, lr, epochs, early_stop_num, verbose=True, writer_flag=False):
    model = model.to(device)
    criterion = nn.MSELoss(reduction='mean')
    params = model.parameters()
    theta1 = model.theta1
    theta2 = model.theta2
    theta3 = model.theta3
    theta4 = model.theta4

    water_model = water_model.to(device)
    air_model = air_model.to(device)

    optimizer = torch.optim.Adam(params, lr=lr)

    create_dir(model_save_path)

    n_epochs, best_loss, step, early_stop_count = epochs, math.inf, 0, 0

    best_loss1 = best_loss2 = best_loss3 = best_loss4 = math.inf
    writer = None
    if writer_flag:
        writer = SummaryWriter()
    ran = range(n_epochs)
    if not verbose:
        ran = tqdm(range(n_epochs), position=0, leave=True)
    for epoch in ran:
        model.train()
        loss_total_record = []
        loss1_record = []
        loss2_record = []
        loss3_record = []
        loss4_record = []

        for x, y1, y2, y3, y4 in train_loader:
            x, y1, y2, y3, y4 = x.to(device), y1.to(device), y2.to(device), y3.to(device), y4.to(device),

            water_expert_represents = water_model(x)[:, water_select_expert, :]
            air_expert_represents = air_model(x)[:, air_select_expert, :]

            pred1, pred2, pred3, pred4 = model(x, water_expert_represents, air_expert_represents)

            loss1 = criterion(pred1, y1)
            loss2 = criterion(pred2, y2)
            loss3 = criterion(pred3, y3)
            loss4 = criterion(pred4, y4)

            loss_total = loss1 / (theta1 ** 2) + loss2 / (theta2 ** 2) + loss3 / (theta3 ** 2) * 1000 + loss4 / (
                        theta4 ** 2) + 2 * (
                                     torch.log(theta1) + torch.log(theta2) + torch.log(theta3) + torch.log(theta4))
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            step += 1

            loss_total_record.append(loss_total.detach().item())
            loss1_record.append(loss1.detach().item())
            loss2_record.append(loss2.detach().item())
            loss3_record.append(loss3.detach().item())
            loss4_record.append(loss4.detach().item())

        mean_train_loss_total = sum(loss_total_record) / len(loss_total_record)
        mean_train_loss1 = sum(loss1_record) / len(loss1_record)
        mean_train_loss2 = sum(loss2_record) / len(loss2_record)
        mean_train_loss3 = sum(loss3_record) / len(loss3_record)
        mean_train_loss4 = sum(loss4_record) / len(loss4_record)
        if writer_flag:
            writer.add_scalar('Loss_total/train', mean_train_loss_total, step)
            writer.add_scalar('Loss1/train', mean_train_loss1, step)
            writer.add_scalar('Loss2/train', mean_train_loss2, step)
            writer.add_scalar('Loss3/train', mean_train_loss3, step)
            writer.add_scalar('Loss4/train', mean_train_loss4, step)

        if verbose and epoch % 100 == 99:
            print(
                f'Epoch [{epoch + 1}/{n_epochs}]: Train loss_total: {mean_train_loss_total:.6f}, loss1: {mean_train_loss1:.6f},loss2: {mean_train_loss2:.6f},loss3: {mean_train_loss3:.6f},loss4: {mean_train_loss4:.6f}')

        if mean_train_loss_total < best_loss:
            best_loss = mean_train_loss_total
            best_loss1 = mean_train_loss1
            best_loss2 = mean_train_loss2
            best_loss3 = mean_train_loss3
            best_loss4 = mean_train_loss4

            torch.save(model.state_dict(), model_save_path)  # Save your best model
            if verbose:
                print(
                    f"\nSave with loss_total: {mean_train_loss_total:.6f}, loss1: {mean_train_loss1:.6f},loss2: {mean_train_loss2:.6f},loss3: {mean_train_loss3:.6f},loss4: {mean_train_loss4:.6f}")
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= early_stop_num:
            print(f'\nModel is not improving, so we halt the training session at epoch: {epoch + 1}.')
            print(
                f"\n Best Model loss_total: {best_loss:.6f}, loss1: {best_loss1:.6f},loss2: {best_loss2:.6f},loss3: {best_loss3:.6f},loss4: {best_loss4:.6f}")
            print(f"\ntheta1:{theta1},theta2:{theta2},theta3:{theta3},theta4:{theta4}")
            return

    print(f'\nTrain all epochs.')
    print(
        f"\n Best Model loss_total: {best_loss:.6f}, loss1: {best_loss1:.6f},loss2: {best_loss2:.6f},loss3: {best_loss3:.6f},loss4: {best_loss4:.6f}")
    print(f"\ntheta1:{theta1},theta2:{theta2},theta3:{theta3},theta4:{theta4}")


def predict(model, device, data, y, water_expert_represents_test, air_expert_represents_test):
    data = torch.Tensor(data).to(device)
    y1 = torch.Tensor(y[:, 0]).to(device)
    y2 = torch.Tensor(y[:, 1]).to(device)
    y3 = torch.Tensor(y[:, 2]).to(device)
    y4 = torch.Tensor(y[:, 3]).to(device)

    criterion = nn.MSELoss(reduction="mean")
    model.eval()
    pred1, pred2, pred3, pred4 = model(data, water_expert_represents_test, air_expert_represents_test)

    pred_list = [pred1, pred2, pred3, pred4]
    rmse1 = criterion(pred1, y1).item() ** 0.5
    rmse2 = criterion(pred2, y2).item() ** 0.5
    rmse3 = criterion(pred3, y3).item() ** 0.5
    rmse4 = criterion(pred4, y4).item() ** 0.5
    rmse_list = [rmse1, rmse2, rmse3, rmse4]

    r1 = r2_score(y1.cpu().detach().numpy(), pred1.cpu().detach().numpy())
    r2 = r2_score(y2.cpu().detach().numpy(), pred2.cpu().detach().numpy())
    r3 = r2_score(y3.cpu().detach().numpy(), pred3.cpu().detach().numpy())
    r4 = r2_score(y4.cpu().detach().numpy(), pred4.cpu().detach().numpy())

    m1 = mean_absolute_error(y1.cpu().detach().numpy(), pred1.cpu().detach().numpy())
    m2 = mean_absolute_error(y2.cpu().detach().numpy(), pred2.cpu().detach().numpy())
    m3 = mean_absolute_error(y3.cpu().detach().numpy(), pred3.cpu().detach().numpy())
    m4 = mean_absolute_error(y4.cpu().detach().numpy(), pred4.cpu().detach().numpy())

    mape1 = mean_absolute_percentage_error(y1.cpu().detach().numpy(), pred1.cpu().detach().numpy())
    mape2 = mean_absolute_percentage_error(y2.cpu().detach().numpy(), pred2.cpu().detach().numpy())
    mape3 = mean_absolute_percentage_error(y3.cpu().detach().numpy(), pred3.cpu().detach().numpy())
    mape4 = mean_absolute_percentage_error(y4.cpu().detach().numpy(), pred4.cpu().detach().numpy())

    r_list = [r1, r2, r3, r4]
    m_list = [m1, m2, m3, m4]
    mape_list = [mape1, mape2, mape3, mape4]
    return rmse_list, r_list, m_list, pred_list, mape_list


##%% md
import numpy as np

##%%
FILE_DIR = "../data/process_data"
MODEL_DIR = "./models"
RESULT_DIR = "./results"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--file_path', type=str, default='data/mutilsolvent2.csv', help='')
    parser.add_argument('--model_save_dir', type=str, default="./p2results_1", help='')
    parser.add_argument('--res_dir', type=str, default="./p2results_1", help='')

    parser.add_argument('--water_model_save_path', type=str, default="water_model_result/waterdi6zhe6.ckpt", help='')
    parser.add_argument('--air_model_save_path', default="air2_model_result/air2di9zhe6.ckpt", type=str, help='')
    parser.add_argument('--water_select_expert_config', default="water_res_result/waterdi6zhe6_top2.csv", type=str,
                        help='')
    parser.add_argument('--air_select_expert_config', type=str, default="air2_res_result/air2di9zhe6_top2.csv", help='')

    parser.add_argument('--input_dim', type=int, default=19, help='')
    parser.add_argument('--represent_dim', type=int, default=100, help='')
    parser.add_argument('--pair_embedding_dim', type=int, default=5, help='')
    parser.add_argument('--expert_num', type=int, default=6, help='')
    parser.add_argument('--gate_output_dim', type=int, default=10, help='')
    parser.add_argument('--epochs', type=int, default=2000, help='')
    parser.add_argument('--early_stop_num', type=int, default=200, help='')

    parser.add_argument('--lr', type=float, default=0.0005, help='')
    parser.add_argument("--verbose", action="store_true", help="")
    parser.add_argument("--writer_flag", action="store_true", help="")
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--test_size', type=float, default=0.2, help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--eps', type=str, default="", help='')
    parser.add_argument("--frozen", action="store_true", help="")

    # device = "cuda:5" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    karg = parser.parse_args(args=[])
    print(karg)
    file_name = os.path.basename(karg.file_path)
    print(f"\n\n\n====start to train {file_name}====")
    model_save_path = os.path.join(karg.model_save_dir, file_name.split(".")[0] + ".ckpt")
    print(karg.file_path)
    create_dir(model_save_path)

    for k in range(0, 15):
        x_train, x_test, y_train, y_test, transfer_X, transfer_y = get_train_test_data(file_path, karg.test_size,
                                                                                       karg.seed, k)

        water_model_save_path = karg.water_model_save_path
        air_model_save_path = karg.air_model_save_path
        model_water = MMoE_Model_Expert()
        model_water.load_state_dict(torch.load(water_model_save_path, map_location=torch.device(device)))
        model_air = MMoE_Model_Expert()
        model_air.load_state_dict(torch.load(air_model_save_path, map_location=torch.device(device)))

        if karg.frozen:
            model_water.pair_embedding_layer_list.requires_grad_(False)
            model_water.expert_list.requires_grad_(False)
            model_water.beta_layer.requires_grad_(False)
            model_air.pair_embedding_layer_list.requires_grad_(False)
            model_air.expert_list.requires_grad_(False)
            model_air.beta_layer.requires_grad_(False)

        water_top2_index_df = pd.read_csv(karg.water_select_expert_config)
        air_top2_index_df = pd.read_csv(karg.air_select_expert_config)

        water_select_expert = torch.Tensor(water_top2_index_df.values).long()
        air_select_expert = torch.Tensor(air_top2_index_df.values).long()

        train_dataset = MMoE_Dataset(x_train, y_train[:, 0], y_train[:, 1], y_train[:, 2], y_train[:, 3])
        train_dataloader = DataLoader(train_dataset, karg.batch_size, shuffle=True, pin_memory=True)
        model = Transfer_Model(input_dim=karg.input_dim, represent_dim=karg.represent_dim,
                               pair_embedding_dim=karg.pair_embedding_dim,
                               expert_num=karg.expert_num, gate_output_dim=karg.gate_output_dim).to(device)

        trainer(model_water, water_select_expert, model_air, air_select_expert, train_dataloader, model,
                model_save_path, device, karg.lr, karg.epochs, karg.early_stop_num, karg.verbose, karg.writer_flag)
        water_expert_represents_test = model_water(torch.Tensor(x_test).to(device))[:, water_select_expert, :]
        air_expert_represents_test = model_air(torch.Tensor(x_test).to(device))[:, air_select_expert, :]
        rmse, r2, m, _, mape = predict(model, device, x_test, y_test, water_expert_represents_test,
                                       air_expert_represents_test)
        print(f"file:{file_name}, rmse:{rmse}, r2:{r2},mae:{m},mape:{mape}")
        result = pd.DataFrame([rmse + r2 + m + mape],
                              columns=["eads_rmse", "delta_e_rmse", "eb_rmse", "db_rmse", "eads_r2", "delta_e_r2",
                                       "eb_r2",
                                       "db_r2", "eads_mae",
                                       "delta_e_mae", "eb_mae", "db_mae", "eads_mape", "delta_e_mape", "eb_mape",
                                       "db_mape"])
        result_path = os.path.join(karg.res_dir, file_name.split(".")[0] + str(k) + "_test.csv")
        create_dir(result_path)
        result.to_csv(result_path, index_label='num')

        water_expert_represents_test = model_water(torch.Tensor(transfer_X).to(device))[:, water_select_expert, :]
        air_expert_represents_test = model_air(torch.Tensor(transfer_X).to(device))[:, air_select_expert, :]
        rmse, r2, m, _, mape = predict(model, device, transfer_X, transfer_y, water_expert_represents_test,
                                       air_expert_represents_test)
        print(f"file:{file_name}, rmse:{rmse}, r2:{r2},mae:{m},mape:{mape}")
        result = pd.DataFrame([rmse + r2 + m + mape],
                              columns=["eads_rmse", "delta_e_rmse", "eb_rmse", "db_rmse", "eads_r2", "delta_e_r2",
                                       "eb_r2",
                                       "db_r2", "eads_mae",
                                       "delta_e_mae", "eb_mae", "db_mae", "eads_mape", "delta_e_mape", "eb_mape",
                                       "db_mape"])
        result_path = os.path.join(karg.res_dir, file_name.split(".")[0] + str(k) + "_transfer.csv")
        create_dir(result_path)
        #
        result.to_csv(result_path, index_label='num')