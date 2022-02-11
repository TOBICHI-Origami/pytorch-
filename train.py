
from model import *
#编写训练代码
def train(train_data_loader , eval_data_loader, model ,optimizer ,num_epoch, log_step_interval, save_step_interval, eval_step_interval, save_path, resume=' '):
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
            ema_loss = 0.9*ema_loss+0.1*bce_loss
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
                ema_eval_loss = 0
                total_acc_acount = 0
                for eval_batch_index,(eval_target,eval_token_index) in enumerate(eval_data_loader):
                    eval_logits = model(eval_token_index)
                    total_acc_acount += (torch.argmax(eval_logits,dim=-1)==eval_target).sum().item()
                    eval_bce_loss = F.binary_cross_entropy(torch.sigmoid(eval_logits),F.one_hot(eval_target,num_classes=2).to(torch.float32))
                    ema_eval_loss = 0.9*ema_eval_loss+0.1*eval_bce_loss
                print("accuracy percentage is {}".format(total_acc_acount/(8*len(eval_data_loader))))
                model.train()


#测试代码
if __name__ == '__main__':
    model = mo()
    print("模型总参数：",sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    train_data_iter = IMDB(root='.deeeeep',split = 'train')
    train_data_loader = torch.utils.data.DataLoader(to_map_style_dataset(train_data_iter),batch_size=BATCH_SIZE,collate_fn = collate_fn,shuffle=True)

    eval_data_iter = IMDB(root="./deeeeeep",split='test')
    eval_data_loader = torch.utils.data.DataLoader(to_map_style_dataset(eval_data_iter),batch_size=8,collate_fn=collate_fn)
    resume = ' '

    train(train_data_loader,eval_data_loader,model,optimizer, num_epoch=10 ,log_step_interval=20,save_step_interval=500,eval_step_interval=100,save_path='./logs_imdb_text_classification',resume=resume)

