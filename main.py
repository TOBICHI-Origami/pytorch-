
#%% [code]
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
import torchtext
import torch
from torch import nn
import torch.nn.functional as F
from torchtext.data import get_tokenizer, to_map_style_dataset
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator

VOCAB_SIZE = 15000
class GCNN(nn.Module):
    def __init__(self,vocab_size=VOCAB_SIZE,embedding_dim = 64, num_class =2):
        super(GCNN,self).__init__()
        self.embedding_table = nn.Embedding(vocab_size,embedding_dim)
        nn.init.xavier_uniform_(self.embedding_table.weight)

        self.conv_A_1 = nn.Conv1d(embedding_dim,64,15,stride=7)
        self.conv_B_1 = nn.Conv1d(embedding_dim, 64, 15, stride=7)

        self.conv_A_2 = nn.Conv1d(64,64,15,stride=7)
        self.conv_B_2 = nn.Conv1d(64,64, 15, stride=7)

        self.output_linear1 = nn.Linear(64,128)
        self.output_linear2 = nn.Linear(128,num_class)

    def forward(self,word_index):
        word_embedding=self.embedding_table(word_index)

        word_embedding = word_embedding.transpose(1,2)
        A = self.conv_A_1(word_embedding)
        B = self.conv_B_1(word_embedding)
        H = A * torch.sigmoid(B)

        A = self.conv_A_2(H)
        B = self.conv_B_2(H)
        H = A * torch.sigmoid(B)

        pool_output = torch.mean(H,dim=-1)
        linear1_output=self.output_linear1(pool_output)
        logits = self.output_linear2(linear1_output)
        return logits
BATCH_SIZE = 64

def yield_token(train_data_iter, tokenizer):
    for i , sample in enumerate(train_data_iter):
        label , comment = sample
        yield tokenizer(comment)
train_data_iter = IMDB(root='./deeeeeep',split='train')
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(yield_token(train_data_iter,tokenizer),min_freq=20,specials=['<unk>'])
vocab.set_default_index(0)
print("单词表大小"+str(len(vocab)))

def collate_fn(batch):
    target = []
    token_index = []
    max_len=0
    for i ,(label,comment) in enumerate(batch):
        tokens = tokenizer(comment)
        token_index.append(vocab(tokens))
        if len(tokens) > max_len:
            max_len = len(tokens)
        if label == 'pos':
            target.append(0)
        else:
            target.append(1)
    token_index = [index+[0]*(max_len-len(index))for index in token_index]
    return (torch.tensor(target).to(torch.int64),torch.tensor(token_index).to(torch.int32))


#编写训练代码
def train(train_data_loader , eval_data_loader, model ,optimizer ,num_epoch,log_step_interval, save_step_interval,eval_step_interval,save_path,resume=' '):
    start_epoch = 0
    start_step =0
    if resume!=' ':
        print("logging from {}".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['optimzer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']

    for epoch_index in range(start_epoch, num_epoch):
        ema_loss=0
        num_batches = len(train_data_loader)

        for batch_index,(target,token_index) in enumerate(train_data_loader):
            optimizer.zero_grad()
            step = num_batches*(epoch_index)+batch_index+1
            logits = model(token_index)
            bce_loss = F.binary_cross_entropy(torch.sigmoid(logits),F.one_hot(target,num_classes=2).to(torch.float32))
            ema_loss=0.9*ema_loss+0.1*bce_loss
            bce_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),0.1)
            optimizer.step()
            if step % log_step_interval==0:
                print("epoch_index:{},batch_index{},ema_loss:{}".format(epoch_index,batch_index,ema_loss))
            if step % save_step_interval ==0:
                os.makedirs(save_path,exist_ok =True)
                save_file = os.path.join(save_path,"step{}.pt".format(step))
                torch.save({
                    'epoch':epoch_index,
                    'step':step,
                    'model_state_dict': optimizer.state_dict(),
                    'loss':bce_loss,
                },save_file)
                print("checkpoint has been saved in {}".format(save_file))
            if step% eval_step_interval==0:
                print("start to do evaluation...")
                model.eval()
                ema_eval_loss=0
                total_acc_acount =0
                total_account = 0
                for eval_batch_index,(eval_target,eval_token_index) in enumerate(eval_data_loader):
                    eval_logits = model(eval_token_index)
                    total_acc_acount +=(torch.argmax(eval_logits,dim=-1)==eval_target).sum().item()
                    eval_bce_loss = F.binary_cross_entropy(torch.sigmoid(eval_logits),F.one_hot(eval_target,num_classes=2).to(torch.float32))
                    ema_eval_loss = 0.9*ema_eval_loss+0.1*eval_bce_loss
                print("accuracy percentage is {}".format(total_acc_acount/(8*len(eval_data_loader))))
                model.train()


#测试代码
if __name__ == '__main__':
    model =GCNN()
    print("模型总参数：",sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    train_data_iter = IMDB(root='.deeeeep',split = 'train')
    train_data_loader = torch.utils.data.DataLoader(to_map_style_dataset(train_data_iter),batch_size=BATCH_SIZE,collate_fn = collate_fn,shuffle=True)

    eval_data_iter = IMDB(root="./deeeeeep",split='test')
    eval_data_loader = torch.utils.data.DataLoader(to_map_style_dataset(eval_data_iter),batch_size=8,collate_fn=collate_fn)
    resume = ' '

    train(train_data_loader,eval_data_loader,model,optimizer, num_epoch=10 ,log_step_interval=20,save_step_interval=500,eval_step_interval=100,save_path='./logs_imdb_text_classification',resume=resume)



