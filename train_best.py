import torch
import torch.utils.data as data
from torchvision import models

import os
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

from dataset import Image_Dataset, Audio_Dataset, ALL_Dataset, one_vis
import argparse
import csv
import glob

# fw = open('RECORD_a.txt', 'w')

# cnt = 0
# one = 0
# zero = 0
# for i in glob.glob('./split_frame_test/*'):
#     video_id = i.split('/')[-1]

#     for j in glob.glob(f'./split_frame_test/{video_id}/*'):
#         if len( glob.glob(f'{j}/*') ) == 0:
#             cnt += 1
#             if int( j[-1] ) == 0 : zero += 1
#             else : one += 1

# print(cnt, one, zero)
# raise

# temp = 0
# for i in glob.glob('./split_frame_test/*'):
#     k = i.split('/')[-1]
#     for j in glob.glob(f'./split_frame_test/{k}/*'):
#         temp += 1
# print(temp)
# raise



print('cuda ? ', torch.cuda.is_available())
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
print('device: ', device)
parser = argparse.ArgumentParser(description='bcnm')
parser.add_argument('--type', type=str, default=None)
parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--vis_ckpt', type=str, default=None)
parser.add_argument('--aud_ckpt', type=str, default=None)
parser.add_argument('--comb_ckpt', type=str, default=None)
parser.add_argument('--train_frame', type=str, default=None)
parser.add_argument('--test_frame', type=str, default=None)
parser.add_argument('--aud_frame', type=str, default=None)
parser.add_argument('--data', type=str, default=None)
parser.add_argument('--out_ckpt', type=str, default=None)
parser.add_argument('--output_path', type=str, default=None)
args = parser.parse_args()



# print( args )
class Classifier_fcn(nn.Module):
    def __init__(self):
        super(Classifier_fcn, self).__init__()
        self.cnn_layers = nn.Sequential(

            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2, 2, 0),

            # nn.Conv2d(32, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.MaxPool2d(2, 2, 0),

            # nn.Conv2d(64, 32, 3, 2, 1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.MaxPool2d(2, 2, 0),

            # nn.Conv2d(32, 16, 3, 2, 1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.MaxPool2d(2, 2, 0),

        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(2, 2, 0),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
    def forward(self, x):
        # print('\nstart: ', x.size())
        x = self.cnn_layers(x)
        # print('-----')
        while (x.size(dim=2) > 1 or x.size(dim=3) > 1):
            # print('before: ', x.size())
            if x.size(dim=2) > 1 and x.size(dim=3) > 1 : x = self.pool(x)
            elif x.size(dim=2) > 1 and x.size(dim=3) == 1:
                pool_a = nn.MaxPool2d([x.size(dim=2), 1], 2, 0)
                x = pool_a(x)
            elif x.size(dim=2) == 1 and x.size(dim=3) > 1:
                pool_b = nn.MaxPool2d([1, x.size(dim=3)], 2, 0)
                x = pool_b(x)
            # print('after: ', x.size())
        # print('-----\n')
        x = x.flatten(0)
        x = x.unsqueeze(0)
        x = self.fc(x)
        # print(x.size())
        return x



class Classifier_Vis(nn.Module):
    def __init__(self):
        super(Classifier_Vis, self).__init__()
        self.cnn_layers = nn.Sequential(

            nn.Conv2d(3, 16, 5, 3, 1), # 3 16 5 3 1
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(16, 32, 3, 2, 1), # 16 32 3 2 1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 64, 3, 1, 1), # 32 64 3 1 1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2, 2, 0)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(6400, 64), # 6400 64
            # nn.ReLU(),
            # nn.Linear(320, 32),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.flatten(1)
        # print(x.shape)
        x = self.fc_layers(x)
        return x



class Classifier_Aud(nn.Module):
    def __init__(self):
        super(Classifier_Aud, self).__init__()
        self.cnn_layers = nn.Sequential(

            nn.Conv2d(2, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.MaxPool2d(3, 32, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(6400, 320),
            nn.ReLU(),
            nn.Linear(320, 2)
        )
    def forward(self, y):
        y = self.cnn_layers(y)
        y = y.flatten(1)
        # print(y.shape)
        y = self.fc_layers(y)
        return y



class Combined_FC(nn.Module):
    def __init__(self):
        super(Combined_FC, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(12800, 128), # 128
            # nn.ReLU(),
            # nn.Linear(178, 16), # 32
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, z):
        z = z.flatten(1)
        z = self.fc_layers(z)
        return z



# model = models.resnet18(pretrained=True)
# model = models.resnet50(pretrained=False)
# model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)

if args.type == 'vis':
    # model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    # model.classifier[-3] = nn.Linear(in_features=4096, out_features=512, bias=True)
    # model.classifier[-1] = nn.Linear(in_features=512, out_features=2, bias=True)
    # model.conv1 = nn.Conv2d(512, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.fc = nn.Linear(model.fc.in_features, 2)
    model = Classifier_Vis()
    # model = Classifier_fcn()
    if os.path.exists(args.vis_ckpt):
        model.load_state_dict(torch.load(args.vis_ckpt, map_location=torch.device(device)))
elif args.type == 'aud':
    model = Classifier_Aud()
    if os.path.exists(args.aud_ckpt):
        model.load_state_dict(torch.load(args.aud_ckpt, map_location=torch.device(device)))
    # model.load_state_dict(torch.load('./ALL_mfcc_dATA_aud_5.ckpt', map_location=torch.device(device)))
elif args.type == 'comb':
    model_a = Classifier_Vis()
    if os.path.exists(args.vis_ckpt):
        model_a.load_state_dict(torch.load(args.vis_ckpt, map_location=torch.device(device)))
    # model_a.load_state_dict(torch.load('./CKPT/A2_vis_feature_30_1e-4_one_vis.ckpt', map_location=torch.device(device)))
    model_a = torch.nn.Sequential( *( list(model_a.children())[:-1] ) )
    model_a = model_a.to(device)

    model_b = Classifier_Aud()
    if os.path.exists(args.aud_ckpt):
        model_b.load_state_dict(torch.load(args.aud_ckpt, map_location=torch.device(device)))
    # model_b.load_state_dict(torch.load('./CKPT/aud_feature_5_1e-3.ckpt', map_location=torch.device(device)))
    model_b = torch.nn.Sequential( *( list(model_b.children())[:-1] ) )
    model_b = model_b.to(device)

    model = Combined_FC()
    if os.path.exists(args.comb_ckpt):
        model.load_state_dict(torch.load(args.comb_ckpt, map_location=torch.device(device)))
    # model.load_state_dict(torch.load('./CKPT/X_comb_comb_25.ckpt', map_location=torch.device(device)))

# model.conv1 = nn.Conv2d(900, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model.fc = nn.Linear(model.fc.in_features, 2)
# model.device = device
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-5) # 1e-4 / 2e-5
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.2, last_epoch = -1)

n_epochs = 30
batch_size = 16
accu = 1
model_path = '???'
best_acc = 0.0

if args.data == 'all':
    train_set = one_vis(data_root_path = args.train_frame, feature_root = args.aud_frame, part = 'all')
elif args.data == 'part':
    train_set = one_vis(data_root_path = args.train_frame, feature_root = args.aud_frame, part = 'part')
train_set_size = int(len(train_set) * 0.9)
valid_set_size = len(train_set) - train_set_size
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size])

test_set = one_vis(data_root_path = args.test_frame, feature_root = args.aud_frame, mode = 'test', part = 'all')

print('train set: ', len(train_set), 'valid set: ', len(valid_set), 'test set: ', len(test_set))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)



# === TRAIN ===
def Train_One_Epoch(model, train_loader, criterion):
    model.train()
    # print(model)

    train_loss = []
    train_accs = []

    for idx, batch in enumerate( tqdm(train_loader) ):
        images, file_names, features = batch

        images = images.to(device)
        features = features.to(device)
        
        labels = []
        for i in range( len(file_names) ):
            labels.append( int( file_names[i].split('_')[-1] ) )
        labels = torch.tensor(labels, dtype=torch.long)

        labels = labels.to(device)

        # print(labels)
        # print(image, file_name)

        # print(images)
        if args.type == 'vis':
            logits = model(images)
        elif args.type == 'aud':
            logits = model(features)
        elif args.type == 'comb':
            feat_vis = model_a(images)
            feat_aud = model_b(features)
            # feat_aud = features

            feat_vis = feat_vis.flatten(1)
            feat_aud = feat_aud.flatten(1)

            comb_feat = torch.cat( [feat_vis, feat_aud], 1 )
            logits = model(comb_feat)

        logits = logits.to(device)
        # print(model.cnn_layers[0].weight.data)
        # print(labels)
        # print(logits)
        # raise BaseException('stop !')

        loss = criterion(logits, labels) # / accu
        optimizer.zero_grad()

        loss.backward()

        # if ( (idx+1)%32 == 0 ):
        #     optimizer.step()
        #     optimizer.zero_grad()
        optimizer.step()

        acc = (logits.argmax(dim=-1) == labels).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)

    scheduler.step()
    print('lr: ', optimizer.param_groups[0]['lr'])
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # === VALID ===
    model.eval()
    valid_accs = []

    for batch in valid_loader:
        images, file_names, features = batch

        images = images.to(device)
        features = features.to(device)
        
        labels = []
        for i in range( len(file_names) ):
            labels.append( int( file_names[i].split('_')[-1] ) )
        labels = torch.tensor(labels, dtype=torch.long)

        labels = labels.to(device)

        if args.type == 'vis':
            logits = model(images)
        elif args.type == 'aud':
            logits = model(features)
        elif args.type == 'comb':
            feat_vis = model_a(images)
            feat_aud = model_b(features)
            # feat_aud = features

            feat_vis = feat_vis.flatten(1)
            feat_aud = feat_aud.flatten(1)

            comb_feat = torch.cat( [feat_vis, feat_aud], 1 )
            logits = model(comb_feat)

        logits = logits.to(device)

        acc = (logits.argmax(dim=-1) == labels).float().mean()
        valid_accs.append(acc)

    valid_acc = sum(valid_accs) / len(valid_accs)
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ], acc = {valid_acc:.5f}")
    global best_acc
    if valid_acc > best_acc:
        best_acc = valid_acc
        print('save best ...')
        # torch.save( model.state_dict(), os.path.join(args.out_ckpt, f'{args.type}_{epoch+1}.ckpt') )
        torch.save( model.state_dict(), os.path.join(args.out_ckpt, f'{args.type}_best.ckpt') )



# === TEST ===
def Test_One_Epoch(model, test_loader):
    with open(args.output_path, 'a') as f:
        header = ['Id', 'Predicted']
        writer = csv.writer(f)
        writer.writerow(header)

    model.eval()
    if args.type == 'comb':
        model_a.eval()
        model_b.eval()
    # print(model)

    TTM = 0
    n_TTM = 0

    for batch in tqdm(test_loader):
        images, file_names, features = batch

        images = images.to(device)
        features = features.to(device)

        if args.type == 'vis':
            logits = model(images)
        elif args.type == 'aud':
            logits = model(features)
        elif args.type == 'comb':
            feat_vis = model_a(images)
            feat_aud = model_b(features)
            # feat_aud = features

            feat_vis = feat_vis.flatten(1)
            feat_aud = feat_aud.flatten(1)

            comb_feat = torch.cat( [feat_vis, feat_aud], 1 )
            logits = model(comb_feat)

        logits = logits.to(device)
        output_label = logits.argmax(dim=-1)

        with open(args.output_path, 'a') as f:
            data = [file_names[0], output_label.item()]
            if output_label.item() == 1 : TTM += 1
            else : n_TTM += 1
            writer = csv.writer(f)
            writer.writerow(data)

    print('TTM: ', TTM)
    print('n_TTM: ', n_TTM)



# === MAIN ===
if args.mode == 'train':
    for epoch in range(n_epochs):
        Train_One_Epoch(model, train_loader, criterion)
        # torch.save(model.state_dict(), f'CKPT/X_{args.type}_comb_{epoch+1+25}.ckpt')
elif args.mode == 'test':
    Test_One_Epoch(model, test_loader)

# fw.close()