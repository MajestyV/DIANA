import numpy as np
import matplotlib.pyplot as plt

class ManyBodySystem_in_Gravity:
    """
    重力多体问题函数包
    This class of function is designed to simulation the trajectories of many-body particle system under the effect of
    gravity.
    """
    def __init__(self,v_init=np.array([[0.01,0.01,0],[-0.05,0,-0.1],[0,-0.01,0]]),
                 q_init=np.array([[-0.5, 0, 0], [0.5, 0, 0], [0,1.0,0]]),
                 num_particle=3,dim_space=3,num_step=500,step_length=0.005):
        # Universal constants
        G = 6.67408e-11  # [=] N-m2/kg2, universal gravitation constant

        # Reference quantities
        m_nd = 1.989e+30  # kg #mass of the sun
        r_nd = 5.326e+12  # m #distance between stars in Alpha Centauri
        v_nd = 30000  # m/s #relative velocity of earth around the sun
        t_nd = 79.91 * 365 * 24 * 3600 * 0.51  # s #orbital period of Alpha Centauri

        self.k1 = G*t_nd*m_nd/(r_nd**2*v_nd)
        self.k2 = v_nd*t_nd/r_nd

        # 系统初始状态
        self.v_init = v_init
        self.q_init = q_init


        self.nbody = num_particle  # 粒子数
        self.dim = dim_space  # 空间维度

        self.m = np.ones(num_particle, dtype=float)  # mass
        self.P = np.empty((num_step, num_particle, dim_space), dtype=float)  # momentum
        self.V = np.empty((num_step, num_particle, dim_space), dtype=float)  # velocity
        self.Q = np.empty((num_step, num_particle, dim_space), dtype=float)  # position, or in other words, trajectory

        self.time = np.empty(num_step, dtype=float)  # 时序信号
        self.nstep = num_step
        self.dt = step_length

    # This function is designed for the calculation of the force constant matrix
    def Interaction(self,q):
        U = np.zeros((self.nbody,self.nbody,self.dim), dtype=float)  # 创建作用势矩阵
        for i in range(self.nbody):
            for j in range(i):
                d_ij = np.linalg.norm(q[i]-q[j])  # 计算质点间的绝对距离
                n_ij = (q[j]-q[i])/d_ij  # 计算左右的方向（归一化基矢）
                u_ij = (self.k1/d_ij**2)*n_ij  # 计算j-th质点对于i-th质点的作用

                U[i,j] = u_ij  # U[i,j]代表了j-th质点对于i-th质点的作用
                U[j,i] = -u_ij  # 反之依然，i-th质点对于j-th质点的作用，应与j-th质点对于i-th质点的作用，大小相等，方向相反

        return U



    # 此函数专用于利用四阶Runge-Kunta方法对系统演化进行数值模拟
    def Evolve_by_RungeKutta(self):
        V, Q, dt, time = (self.V, self.Q, self.dt, self.time)

        # 定义速度矩阵
        v1 = np.zeros((self.nbody,self.dim), dtype=float)  #
        v2 = np.zeros((self.nbody, self.dim), dtype=float)
        v3 = np.zeros((self.nbody, self.dim), dtype=float)
        v4 = np.zeros((self.nbody, self.dim), dtype=float)
        # 定义加速度矩阵
        a1 = np.zeros((self.nbody, self.dim), dtype=float)  #
        a2 = np.zeros((self.nbody, self.dim), dtype=float)
        a3 = np.zeros((self.nbody, self.dim), dtype=float)
        a4 = np.zeros((self.nbody, self.dim), dtype=float)
        # 定义位置矩阵
        q1 = np.zeros((self.nbody, self.dim), dtype=float)  #
        q2 = np.zeros((self.nbody, self.dim), dtype=float)
        q3 = np.zeros((self.nbody, self.dim), dtype=float)
        q4 = np.zeros((self.nbody, self.dim), dtype=float)

        # 系统初始化
        t = 0
        v = self.v_init
        q = self.q_init

        for n in range(self.nstep):
            # 保存轨迹到结果数组
            time[n] = t
            V[n] = v
            Q[n] = q

            # 计算一阶R-K结果
            v1 = v
            q1 = q
            U1 = self.Interaction(q1)
            for i in range(self.nbody):
                a1[i] = 0
                for j in range(self.nbody):
                    a1[i] += U1[i,j]*self.m[i]

            # 计算二阶R-K结果
            v2 = v1 + a1*dt/2.0
            q2 = q1 + self.k2*v1*dt/2.0
            U2 = self.Interaction(q2)
            for i in range(self.nbody):
                a2[i] = 0
                for j in range(self.nbody):
                    a2[i] += U2[i, j] * self.m[i]

            # 计算三阶R-K结果
            v3 = v1 + a2 * dt / 2.0
            q3 = q1 + self.k2*v2 * dt / 2.0
            U3 = self.Interaction(q3)
            for i in range(self.nbody):
                a3[i] = 0
                for j in range(self.nbody):
                    a3[i] += U3[i, j] * self.m[i]

            # 计算四阶R-K结果
            v4 = v1 + a3 * dt
            q4 = q1 + self.k2*v3 * dt
            U4 = self.Interaction(q4)
            for i in range(self.nbody):
                a4[i] = 0
                for j in range(self.nbody):
                    a4[i] += U4[i, j] * self.m[i]

            # 计算下一时刻的轨迹
            t = t + dt
            v = v + dt*(a1+2*a2+2*a3+a4)/6.0
            q = q + dt*self.k2*(v1+2*v2+2*v3+v4)/6.0

        return time, V, Q

#A function defining the equations of motion
#def TwoBodyEquations(w,t,G,m1,m2):
    #r1=w[:3]
    #r2=w[3:6]
    #v1=w[6:9]
    #v2=w[9:12]
    #r=sci.linalg.norm(r2-r1) #Calculate magnitude or norm of vector
    #dv1bydt=K1*m2*(r2-r1)/r**3
    #dv2bydt=K1*m1*(r1-r2)/r**3
    #dr1bydt=K2*v1
    #dr2bydt=K2*v2
    #r_derivs=sci.concatenate((dr1bydt,dr2bydt))
    #derivs=sci.concatenate((r_derivs,dv1bydt,dv2bydt))
    #return derivs

if __name__ == "__main__":
    # r = np.array([[-0.5, 0,0], [0.5, 0,0]])
    # v = np.array([[0.01,0.01,0],[-0.05,0,-0.1]])
    r = np.array([[-0.5, 0, 0], [0.5, 0, 0], [0,1.0,0]])
    v = np.array([[0.01,0.01,0],[-0.05,0,-0.1],[0,-0.01,0]])

    MB = ManyBodySystem_in_Gravity(v_init=v,q_init=r,num_particle=3,dim_space=3,num_step=500,step_length=0.005)
    t, vel, pos = MB.Evolve_by_RungeKutta()

    # 画图模块
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    # Create 3D axes
    ax = fig.add_subplot(111, projection="3d")
    # Plot the orbits
    ax.plot(pos[:,0,0],pos[:,0,1],pos[:,0,2], color="darkblue")
    ax.plot(pos[:,1,0],pos[:,1,1],pos[:,1,2], color="tab:red")
    ax.plot(pos[:,2,0],pos[:,2,1],pos[:,2,2], color='g')
    # Plot the final positions of the stars
    ax.scatter(pos[-1,0,0],pos[-1,0,1],pos[-1,0,2], color="darkblue", marker="o", s=100, label="Alpha Centauri alpha")
    ax.scatter(pos[-1,1,0],pos[-1,1,1],pos[-1,1,2], color="tab:red", marker="o", s=100, label="Alpha Centauri beta")
    ax.scatter(pos[-1,2,0],pos[-1,2,1],pos[-1,2,2], color="g", marker="o", s=100, label="Third star")
    # Add a few more bells and whistles
    ax.set_xlabel("x-coordinate", fontsize=14)
    ax.set_ylabel("y-coordinate", fontsize=14)
    ax.set_zlabel("z-coordinate", fontsize=14)
    ax.set_title("Visualization of orbits of stars in a three-body system\n", fontsize=14)
    ax.legend(loc="upper left", fontsize=14)

    plt.show(block=True)


    #fig = plt.figure()
    #ax = fig.add_subplot(projection="3d")  # 设置画布为三维图图像
    #ax.plot(pos[:,0,0],pos[:,0,1],pos[:,0,2])
    #ax.plot(pos[:, 1, 0], pos[:, 1, 1],pos[:,1,2])
    #ax.plot(pos[:, 2, 0], pos[:, 2, 1], pos[:, 2, 2])

    #plt.show(block=True)

    # V = MB.Interaction(r)

    # print(V[1,0])