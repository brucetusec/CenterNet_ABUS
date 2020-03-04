import torch
import torch.nn as nn
import time

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self)) # 模型的默认名字

    def load(self, path, ep):
        #path = path + self.model_name + '_' + str(ep)
        path =  path + 'hourglass_' + str(ep)
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        prefix = 'checkpoints/' + 'hourglass_'
        '''
        e.g. AlexNet_0710_23:57:29.pth
        '''
        if name is None:
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        else:
            name = prefix + name
            
        torch.save(self.state_dict(), name)
        return name