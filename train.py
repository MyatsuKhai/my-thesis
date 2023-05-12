import argparse
import torch 
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def arg_parser():
    parser = argparse.ArgumentParser(description = 'train.py')
    parser.add_argument('--arch', dest='arch', action='store',default ='vgg16',type = str)
    parser.add_argument('--save_dir',dest='save_dir',action ='store',default ='final_checkpoint.pth')
    parser.add_argument('--learning_rate', dest ='learning_rate',action='store',default ='0.001', type = float)
    parser.add_argument('--hidden_units', dest ='hidden_units',action ='store',default = '512',type=int)
    parser.add_argument('--epochs',dest = 'epochs',action = 'store',default ='30',type=int)
    parser.add_argument('--gpu',dest ='gpu',action = 'store',default ='gpu')

    args = parser.parse_args()
    return args

def train_transform(train_dir):
    transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                                [0.229,0.224,0.225])])
    train_dataset = datasets.ImageFolder(train_dir,transform=transform)
    return train_dataset

def test_transform(test_dir):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                                [0.229,0.224,0.225])])
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    return test_dataset

def data_loader (data,train = True):
    dataloader = torch.utils.data.DataLoader(data,batch_size = 30,shuffle = True)
    return dataloader

def initial_model(architecture):
    model = models.vgg16(pretrained = True)
    model.name = 'vgg16'
    print(model.name)
    for param in model.parameters():
        param.requires_grad = False
    return model

def initial_classifier(model,hidden_units):
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('fc1',nn.Linear(25088,2000)),
        ('relu1',nn.ReLU()),
        ('dropout',nn.Dropout(0.3)),
        ('fc2',nn.Linear(2000,1000)),
        ('relu2',nn.ReLU()),
        ('dropout',nn.Dropout(0.3)),
        ('fc3',nn.Linear(1000,30)),
        ('output',nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    print(model)
    print("No. of hidden units(fc1) :" , hidden_units)
    return classifier

def gpu_checking (gpu_arg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        print("CUDA is not found, CPU is used")
    return device

def network_training (model,trainloader,validloader):
    epoch = 50
    running_loss = 0
    print("Number of epochs :", epoch)
    print('Training process initializing...\n')
    for e in range(epoch):
        for images,labels in trainloader:
            steps += 1
            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(images)
            loss = criterion(logps,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                valid_loss,accuracy = 0,0
                model.eval()
                with torch.no_grad():
                    for images,labels in validloader:
                        images,labels = images.to(device), labels.to(device)
                        output = model.forward(images)
                        batch_loss = criterion(output,labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(output)
                        top_p , top_class = ps.topk(1,dim =1)
                        equals = top_class ==labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {e+1}/{epoch}.."
                    f"Train loss : {running_loss/print_every:.3f}.."
                    f"Valid loss: {valid_loss/len(validloader):.3f}.."
                    f"Valid accuracy: {accuracy/len(validloader):.3f}..")
                running_loss = 0
                model.train()
    return model


def network_validation (model,test_loader,device):
    total, accuracy = 0,0
    model.eval()
    with torch.no_grad():
        for images,labels in test_loader:
            images,labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    print('Testing Accuracy is : %d%%' % (100 * accuracy / total))
    
    
def initial_checkpoint (model,save_dir,train_data,optimizer):
    model.class_to_idx = train_data.class_to_idx
    torch.save({'convolutional base':'vgg16',
                'fc1': 2000,
                'dropout': 0.3,
                'epochs': 50,
                'state_dict': model.state_dict(),
                'class_to_idx' : model.class_to_idx,
                'classifier' : model.classifier,
                'optimizer_dict': optimizer.state_dict()}, 'final_checkpoint.pth')
            
def main():
    args = arg_parser()
    data_dir = 'goods'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_data = train_transform(train_dir)
    valid_data = test_transform(valid_dir)
    test_data = test_transform(test_dir)

    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data,train = False)
    testloader = data_loader(test_data, train=False)
    print(len(trainloader))

    model = initial_model(args.arch)

    model.classifier = initial_classifier(model,hidden_units=args.hidden_units)

    device = gpu_checking(gpu_arg=args.gpu);
    model.to(device);

    learning_rate = args.learning_rate
    print('Learning rate specified as:', learning_rate)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)
    print_every = 1
    steps = 0

    trained_model = network_training(model,trainloader,validloader,device,criterion,optimizer,args.epochs,print_every,steps)
    print("Training is complected!")

    network_validation(trained_model,testloader,device)

    initial_checkpoint(trained_model,args.save_dir,train_data,optimizer)
if __name__ == '__main__': main()