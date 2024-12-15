import numpy as np
from simulator import Simulator
from pathlib import Path
from typing import Dict
import pinocchio as pin
import os
import scipy
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
model = pin.buildModelFromMJCF(xml_path)
data = model.createData()

class RobotController:
    def __init__(self):
        self.Kp = np.diag([86, 86, 86, 86, 86, 86]) 
        self.Kd = np.diag([20, 30, 30, 20, 20, 20]) 
        
        self.mode = "not" #режим работы робота
        
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
        self.model = pin.buildModelFromMJCF(xml_path)
        self.data = model.createData()
        
    def skew_to_vector(self, screw):
        return np.array([[screw[2, 1]],
                         [screw[0, 2]],
                         [screw[1, 0]]],dtype=float)
    
    def so3_error(self, Rd, R):
        error = Rd @ R.T
        error_log = scipy.linalg.logm(error)
        error_vector = self.skew_to_vector(error_log)
        return error_vector
            
    def circular_trajectory(self,t, center, rad, w):
        x_c, y_c, z_c = center

        x = x_c + rad * np.cos(w * t)
        y = z_c + rad * np.sin(w * t)
        z = z_c

        vx = -rad * w * np.sin(w * t)
        vy = rad * w * np.cos(w * t)
        vz = 0

        ax = -rad * w**2 * np.cos(w * t)
        ay = -rad * w**2 * np.sin(w * t)
        az = 0
        
        position = np.array([x, y, z]).reshape(-1, 1)  #здесь и далее для преобразования в вектор
        velocity = np.array([vx, vy, vz, 0, 0, 0]).reshape(-1, 1) 
        acceleration = np.array([ax, ay, az, 0, 0, 0]).reshape(-1, 1) 
        
        return position, velocity, acceleration    
    
    def plot_results(self,times: np.ndarray, positions: np.ndarray, 
                     velocities: np.ndarray, control: np.ndarray, eef_pos: np.ndarray):
        
        plt.figure(figsize=(10, 6))
        for i in range(positions.shape[1]):
            plt.plot(times, positions[:, i], label=f'Joint {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Joint Positions [rad]')
        plt.title('Joint Positions')
        plt.legend()
        plt.grid(True)
        plt.savefig('logs/plots/positions.png')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        for i in range(velocities.shape[1]):
            plt.plot(times, velocities[:, i], label=f'Joint {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Joint Velocities [rad/s]')
        plt.title('Joint Velocities')
        plt.legend()
        plt.grid(True)
        plt.savefig('logs/plots/velocities.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        for i in range(control.shape[1]):
            plt.plot(times, control[:, i], label=f'Joint {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Control')
        plt.title('Control')
        plt.legend()
        plt.grid(True)
        plt.savefig('logs/plots/controls.png')
        plt.close()    
        
        plt.figure(figsize=(10, 6))
        for i in range(eef_pos.shape[1]):
            plt.plot(times, eef_pos[:, i], label=['x','y','z'])
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title('End Effector position')
        plt.legend()
        plt.grid(True)
        plt.savefig('logs/plots/eef_pos.png')
        plt.close()   
    
    def task_space_controller(self, q: np.ndarray, dq: np.ndarray, t: float, desired: Dict) -> np.ndarray:

        if self.mode == "trajectory":
            # параметры окружностей
            center = [0.0, 0.9, 0.5]  
            rad = 0.2             
            w = 2 * np.pi / 1    

            roll = np.deg2rad(0)
            pitch = np.deg2rad(0)
            yaw = np.deg2rad(0)
            
            p_des, dp_des, ddp_des = self.circular_trajectory(t, center, rad, w)
            R_des = pin.utils.rpyToMatrix(roll, pitch, yaw)

        else:
            if self.mode == "follow":
                p_des = desired['pos'] # уставка по х у z 

                desired_quaternion = desired['quat'] # это надо что бы муждоко понял 

                # преобразование в матрицу SE3
                desired_quaternion_pin = np.array([*desired_quaternion[1:], desired_quaternion[0]]) 
                desired_pose = np.concatenate([p_des, desired_quaternion_pin])

                R_des = pin.XYZQUATToSE3(desired_pose).rotation
                p_des = p_des.reshape(-1, 1)

            else:
                # иначе для заданных через p_des координат
                p_des = np.array([[0.3],         
                                  [-0.3],       
                                  [0.4],          
                                 ],dtype=float)

                roll = np.deg2rad(0)
                pitch = np.deg2rad(0)
                yaw = np.deg2rad(0)

                R_des = pin.utils.rpyToMatrix(roll, pitch, yaw)


            dp_des = np.array([[0],          # линейная скорость
                            [0],             
                            [0],             
                            [np.deg2rad(0)], # угловая скорость
                            [np.deg2rad(0)], 
                            [np.deg2rad(0)], 
                            ],dtype=float)


            ddp_des = np.array([[0],             # линейное ускорение
                                [0],            
                                [0],             
                                [np.deg2rad(0)], # угловое ускорение
                                [np.deg2rad(0)], 
                                [np.deg2rad(0)], 
                                ],dtype=float)


        # Составляющие ПД регулятора
        Kp = np.diag([86, 86, 86, 86, 86, 86]) 

        Kd = np.diag([20, 30, 30, 20, 20, 20]) 

       
        
        pin.computeAllTerms(model, data, q, dq)
        pin.forwardKinematics(model, data, q, dq)

        ee_frame_id = model.getFrameId("end_effector")
        frame = pin.LOCAL

        pin.updateFramePlacement(model, data, ee_frame_id)
        ee_pose = data.oMf[ee_frame_id]
        p = ee_pose.translation.reshape(-1, 1) 
        R = ee_pose.rotation

        # тут считаются скорости 
        twist = pin.getFrameVelocity(model, data, ee_frame_id, frame)
        v = twist.linear.reshape(-1, 1) 
        w = twist.angular.reshape(-1, 1) 
        dp = np.vstack((v, w)) 

        desired_twist = ee_pose.actInv(pin.Motion(dp_des.flatten())) 
        dp_des_local = np.hstack([desired_twist.linear, desired_twist.angular]).reshape(-1, 1) 

       
        M = data.M

        h = data.nle

        J = pin.getFrameJacobian(model, data, ee_frame_id, frame) # Якобиан для ур-я ЭЛ

        pin.computeJointJacobiansTimeVariation(model, data, q, dq)
        dJ = pin.getFrameJacobianTimeVariation(model, data, ee_frame_id, frame) # Дифф Якобиана

        desired_acc = ee_pose.actInv(pin.Motion(ddp_des[:3], ddp_des[3:]))
        ddp_des_local = np.hstack([desired_acc.linear, desired_acc.angular]).reshape(-1, 1)

        p_err = p_des - p #ошибки
        rot_err = self.so3_error(R_des, R)
        pose_err = np.vstack((p_err, rot_err))
        dp_err = dp_des_local - dp 

        #управление
        ddp = Kp @ pose_err + Kd @ dp_err + ddp_des_local

        ddq_des = np.linalg.pinv(J) @ (ddp - dJ @ dq.reshape(-1, 1))

        control = M @ ddq_des + h.reshape(-1, 1)
        return control.flatten()

def main():
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning task space controller...")
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=True,
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/task_space.mp4",
        fps=30,
        width=1920,
        height=1080
    )
    
    robot = RobotController()
    
    sim.set_controller(robot.task_space_controller)

    if True:
        t = 0
        dt = sim.dt
        time_limit = 10.0
        
        tim = []
        positions = []
        velocities = []
        controls = []
        eef_positions = []
        
        while t < time_limit:
            state = sim.get_state()
            tim.append(t)
            positions.append(state['q'])
            velocities.append(state['dq'])
            eef_positions.append(state["ee_pos"])
            
            tau = robot.task_space_controller(q=state['q'], dq=state['dq'], t=t, desired={"pos":np.array([0,0.5,0.5]), "quat":np.array([0,0,0,0])})
            controls.append(tau)

            sim.step(tau)
            
            t += dt
        
        tim = np.array(tim)
        positions = np.array(positions)
        velocities = np.array(velocities)
        controls = np.array(controls)
        eef_positions = np.array(eef_positions)

        robot.plot_results(tim, positions, velocities, controls,eef_positions)


    sim.run(time_limit=2.0)

if __name__ == "__main__":
    main() 