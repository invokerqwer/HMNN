import argparse
import math
import os
from sklearn.model_selection import KFold
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import KFold
from datasets import get_train_test_data, MMoE_Dataset
from models import MMoE_Model
import numpy as np
import random

def create_dir(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))



def trainer(train_loader, model, model_save_path, device ,lr, epochs, early_stop_num,verbose=True, writer_flag=False):

    criterion = nn.MSELoss(reduction='mean')


    params = model.parameters()
    theta1 = model.theta1
    theta2 = model.theta2
    theta3 = model.theta3
    theta4 = model.theta4

    similarity_criterion = nn.CosineSimilarity()


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
            optimizer.zero_grad()
            x, y1, y2, y3, y4 = x.to(device), y1.to(device), y2.to(device), y3.to(device), y4.to(device),

            pred1, pred2, pred3, pred4,s1,s2,s3,s4,_ = model(x)

            loss1 = criterion(pred1, y1)
            loss2 = criterion(pred2, y2)
            loss3 = criterion(pred3, y3)
            loss4 = criterion(pred4, y4)

            loss5 = similarity_criterion(s1, s2) + similarity_criterion(s1, s3) + similarity_criterion(s1, s4) \
                    + similarity_criterion(s2, s3) + similarity_criterion(s2, s4) + similarity_criterion(s3, s4)
            loss5 = loss5.sum()
            loss5.backward(retain_graph=True)
            loss_total = loss1 / (theta1 ** 2) +  loss2 / (theta2 ** 2) + loss3 / (theta3 ** 2) + loss4 / (theta4 ** 2) +  2 * (torch.log(theta1) +torch.log(theta2) + torch.log(theta3) + torch.log(theta4))



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




def predict(model,device,data,y):
    data = torch.Tensor(data).to(device)
    y1 = torch.Tensor(y[:,0]).to(device)
    y2 = torch.Tensor(y[:,1]).to(device)
    y3 = torch.Tensor(y[:,2]).to(device)
    y4 = torch.Tensor(y[:,3]).to(device)

    criterion = nn.MSELoss(reduction="mean")
    model.eval()
    pred1,pred2,pred3,pred4,_,_,_,_,_,= model(data)

    pred_list = [pred1,pred2,pred3,pred4]
    rmse1  = criterion(pred1,y1).item()**0.5
    rmse2  = criterion(pred2,y2).item() ** 0.5
    rmse3  = criterion(pred3,y3).item()** 0.5
    rmse4  = criterion(pred4,y4).item()** 0.5
    rmse_list = [rmse1,rmse2,rmse3,rmse4]

    r1 = r2_score(y1.cpu().detach().numpy(),pred1.cpu().detach().numpy())
    r2 = r2_score(y2.cpu().detach().numpy(),pred2.cpu().detach().numpy())
    r3 = r2_score(y3.cpu().detach().numpy(),pred3.cpu().detach().numpy())
    r4 = r2_score(y4.cpu().detach().numpy(),pred4.cpu().detach().numpy())

    m1 = mean_absolute_error(y1.cpu().detach().numpy(),pred1.cpu().detach().numpy())
    m2 = mean_absolute_error(y2.cpu().detach().numpy(),pred2.cpu().detach().numpy())
    m3 = mean_absolute_error(y3.cpu().detach().numpy(),pred3.cpu().detach().numpy())
    m4 = mean_absolute_error(y4.cpu().detach().numpy(),pred4.cpu().detach().numpy())

    mape1 = mean_absolute_percentage_error(y1.cpu().detach().numpy(),pred1.cpu().detach().numpy())
    mape2 = mean_absolute_percentage_error(y2.cpu().detach().numpy(),pred2.cpu().detach().numpy())
    mape3 = mean_absolute_percentage_error(y3.cpu().detach().numpy(),pred3.cpu().detach().numpy())
    mape4 = mean_absolute_percentage_error(y4.cpu().detach().numpy(),pred4.cpu().detach().numpy())

    r_list = [r1,r2,r3,r4]
    m_list = [m1,m2,m3,m4]
    mape_list = [mape1,mape2,mape3,mape4]
    return rmse_list,r_list,m_list,pred_list,mape_list




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--file_path', type=str, default='data/air2.csv',help='')
    parser.add_argument('--model_save_dir', type=str,default='air2_model_result', help='')
    parser.add_argument('--res_dir', type=str,default='air2_res_result', help='')



    parser.add_argument('--input_dim', type=int,default=19, help='')
    parser.add_argument('--represent_dim', type=int, default=100, help='')
    parser.add_argument('--pair_embedding_dim', type=int, default=5, help='')
    parser.add_argument('--expert_num', type=int, default=5, help='')
    parser.add_argument('--epochs',  type=int, default=2000, help='')
    parser.add_argument('--early_stop_num',  type=int, default=200, help='')

    parser.add_argument('--lr', type=float, default=0.0005, help='')
    parser.add_argument("--verbose", action="store_true", help="")
    parser.add_argument("--writer_flag", action="store_true", help="")
    parser.add_argument('--batch_size',  type=int, default=128, help='')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='')
    parser.add_argument('--seed',  type=int, default=42, help='')


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    karg = parser.parse_args()


    file_name = os.path.basename(karg.file_path)
    print(f"\n\n\n====start to train {file_name}====")
    #接下来在这里在套一个外层大循环，这和循环直接执行10次
    for p in range(10):
        rand_num = random.randint(0, 100)
        kf = KFold(n_splits=10,shuffle=True,random_state=42)
        #在此处我将完成交叉验证10次的选取,接下来我将保存结果，将所有数据都报错起来
        or_data = pd.read_csv(karg.file_path).values
        k_num=0
        for train_index , test_index in kf.split(or_data):  # 调用split方法切分数据
            x_train, x_test, y_train, y_test = get_train_test_data(karg.file_path, train_index , test_index)
            train_dataset = MMoE_Dataset(x_train, y_train[:, 0], y_train[:, 1], y_train[:, 2], y_train[:, 3])
            train_dataloader = DataLoader(train_dataset, karg.batch_size, shuffle=True, pin_memory=True)
            model_save_path =  os.path.join(karg.model_save_dir,file_name.split(".")[0]+"di"+str(k_num)+"zhe"+str(p)+".ckpt")
            create_dir(model_save_path)
            model = MMoE_Model(input_dim=karg.input_dim, represent_dim=karg.represent_dim, pair_embedding_dim=karg.pair_embedding_dim,
                               expert_num=karg.expert_num).to(device)

            trainer(train_dataloader, model, model_save_path, device, karg.lr, karg.epochs, karg.early_stop_num, karg.verbose, karg.writer_flag)
            rmse, r2, m, _, mape = predict(model, device, x_train, y_train)
            print("=====")
            print(mape)
            rmse, r2, m, _, mape = predict(model, device, x_test, y_test)
            print(f"file:{file_name}, rmse:{rmse}, r2:{r2},mae:{m},mape:{mape}")

            result = pd.DataFrame([rmse + r2 + m + mape],
                                  columns=["eads_rmse", "delta_e_rmse", "eb_rmse", "db_rmse", "eads_r2", "delta_e_r2", "eb_r2",
                                           "db_r2", "eads_mae",
                                           "delta_e_mae", "eb_mae", "db_mae", "eads_mape", "delta_e_mape", "eb_mape",
                                           "db_mape"])

            pred1, pred2, pred3, pred4, s1, s2, s3, s4, gates = model(torch.Tensor(x_train).to(device))
            weight = pd.DataFrame(gates.cpu().mean(1).squeeze(-1).detach().numpy())

            top2_index_list = []
            for i in range(4):
                top2 = np.sort(weight.iloc[i])[-2:]
                top2_index = np.argsort(weight.iloc[i])[-2:]
                top2_index_list.append(top2_index.values)
            top2_index_df = pd.DataFrame((top2_index_list))
            top2_index_df.columns = ["top2", "top1"]


            result_path = os.path.join(karg.res_dir, file_name.split(".")[0]+"di"+str(k_num)+"zhe"+str(p)+".csv")
            top2_path = os.path.join(karg.res_dir, file_name.split(".")[0]+"di"+str(k_num)+"zhe"+str(p)+"_top2.csv")
            create_dir(result_path)
            top2_index_df.to_csv(top2_path, index=False)
            result.to_csv(result_path, index_label='num')
            k_num=k_num+1