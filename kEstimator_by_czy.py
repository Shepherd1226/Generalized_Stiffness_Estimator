import numpy as np
from sklearn.linear_model import LinearRegression

class DexHand_kEstimator():
    def __init__(self):
        self.datanum=30
        self.forcelist=np.array([])
        self.poslist=np.array([])
        self.last_goal_force=None
        self.last_pos=None
        self.minpos=None
        self.maxpos=None
        self.gamma=0.995#0.995 performs best in real tests
        self.last_k=0.05
        self.k=0.05
        self.k_out=self.k

    def data_push_in(self,nowpos,nowforce,goalforce):
        nowforce=goalforce
        # 更新数据
        if len(self.forcelist) < self.datanum:
            self.forcelist = np.append(self.forcelist, nowforce)
            self.poslist = np.append(self.poslist, nowpos)
        else:
            # 舍弃屈服影响太大导致的数据点
            if (self.last_goal_force):
                if abs(goalforce-self.last_goal_force)<0.002:
                   self.poslist+=nowpos-(self.last_pos+(goalforce-self.last_goal_force)/self.k)
                #    self.last_goal_force=goalforce
                #    self.last_pos=nowpos
                #    return
            self.last_goal_force=goalforce
            self.last_pos=nowpos

            # 找到最小和最大横坐标
            minpos = np.min(self.poslist)
            maxpos = np.max(self.poslist)

            if nowpos < minpos:
                # 替换掉原来横坐标最大的点
                idx_to_replace = np.argmax(self.poslist)
            elif nowpos > maxpos:
                # 替换掉原来横坐标最小的点
                idx_to_replace = np.argmin(self.poslist)
            else:
                # 替换掉差距最小的点
                idx_to_replace = np.argmin(np.abs(self.poslist - nowpos))

            # 更新数组
            self.forcelist[idx_to_replace] = nowforce
            self.poslist[idx_to_replace] = nowpos

        # 进行线性拟合
        if(len(self.poslist)>2):
            # 创建线性回归模型
            model = LinearRegression()

            # 将数据转换为适当的形状
            X = np.array(self.poslist).reshape(-1, 1)
            y = np.array(self.forcelist)

            # 拟合模型
            model.fit(X, y)

            # 获取斜率
            slope = model.coef_[0]
            self.k = self.gamma*self.last_k+slope*(1-self.gamma)  # 斜率
            self.last_k=self.k

        if self.k<0.0005:
            self.k = 0.0005
        elif self.k>0.8:
            self.k=0.8

        self.k_out=self.k
        # self.k = 0.05