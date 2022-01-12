from time import sleep
import pybullet as p
import numpy as np
import cv2
import time
import pybullet_utils.bullet_client as bullet_client

class Environment:
    def __init__(self) -> None:
        '''
            Setting up environment constructor
        '''
        self.guiClient = bullet_client.BulletClient(p.GUI)
        self.dirClient = bullet_client.BulletClient(p.DIRECT)

        self.dirClient.setAdditionalSearchPath("/data/")
        self.guiClient.setAdditionalSearchPath("/data/")
        self.guiClient.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True)
        
        self.guiClient.loadURDF("table/table.urdf", [-0.2, -0.2, -1], useFixedBase=True)
        
        self.guiClient.loadURDF("basket/basket.urdf", [-1, 0, -1])
        self.guiClient.loadURDF("basket/basket.urdf", [-1, -0.4, -1])
        self.guiClient.loadURDF("basket/basket.urdf", [-1, 0.4, -1])

        self.dirClient.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True)
        
        self.dirClient.loadURDF("table/table.urdf", [-0.2, -0.2, -1], useFixedBase=True)
        
        self.dirClient.loadURDF("basket/basket.urdf", [-1, 0, -1])
        self.dirClient.loadURDF("basket/basket.urdf", [-1, -0.4, -1])
        self.dirClient.loadURDF("basket/basket.urdf", [-1, 0.4, -1])
        self.guiClient.setGravity(0,0,-10) 
        self.dirClient.setGravity(0,0,-10) 

        self.box_position= None;
        self.box_color = None;


    def get_camera_image(self):
        view_matrix = self.guiClient.computeViewMatrix(
            cameraEyePosition=[0, 0, 0.98],
            cameraTargetPosition=[-0.2, -0.2, .35],
            cameraUpVector=[0, 1, 0])

        projection_matrix = self.guiClient.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=1.5)
            
        width, height, rgb, depth, seg = self.guiClient.getCameraImage(
            width=480,
            height=320,
            projectionMatrix = projection_matrix,
            viewMatrix = view_matrix
        )
        return rgb
    def detection(self):
        global green_pos
        global color
        
        frame = self.get_camera_image()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        lower_green = np.array([50, 100, 100])
        upper_green = np.array([70, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        lower_red = np.array([0, 255, 116])
        upper_red = np.array([6, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        count_blue = cv2.countNonZero(blue_mask)
        count_green = cv2.countNonZero(green_mask)
        count_red = cv2.countNonZero(red_mask)
        
        if count_green > 0:
            self.box_color = "Green"
        
        elif count_red > 0:
            print("KIRMIZI" )
            self.box_color = "Red"
            
        elif count_blue > 0:
            print("MAVI" )
            self.box_color = "Blue"
            
        res = cv2.bitwise_and(frame, frame, mask= green_mask)
        res2 = cv2.bitwise_and(frame, frame, mask= red_mask)
        res3 = cv2.bitwise_and(frame, frame, mask= blue_mask)
        # 
        # cv2.imshow("frame", frame)
        # cv2.imshow("green", res)
        # cv2.imshow("red", res2)
        # cv2.imshow("blue", res3)
    def randomSpawn(self):

        arr = np.random.randint(3,size=1)
        arr1 = np.random.randint(300,size=2)
        arr1+=50
        x = -arr1[0]/1000
        y = -arr1[1]/1000


        box_id = 0
        # box_id = p.loadURDF("cube_green.urdf", [x, y, -0.35],globalScaling=0.42)
        if arr[0]==0:
            box_id = self.guiClient.loadURDF("cube_blue.urdf", [x, y, -0.35],globalScaling=0.42)
        elif arr[0]==1:
            box_id = self.guiClient.loadURDF("cube_red.urdf", [x, y, -0.35],globalScaling=0.42)
        elif arr[0]==2:
            box_id = self.guiClient.loadURDF("cube_green.urdf", [x, y, -0.35],globalScaling=0.42)

        self.box_position, cubeOrn = self.guiClient.getBasePositionAndOrientation(box_id)
    def runNSteps(self,n):
        for _ in range(n):
            self.guiClient.stepSimulation();
        self.get_camera_image()
    
class Robot:
    def __init__(self,client) -> None:
        self.client = client
        self.baxterId= self.client.loadURDF("baxter_common/baxter_description/urdf/toms_baxter.urdf", 
                                            useFixedBase=True)
        self.client.resetBasePositionAndOrientation(self.baxterId, [0.2, -0.8, 0.0], [0., 0., -1., -1.])
        
    def setMotors(self,jointPoses):
        numJoints = p.getNumJoints(self.baxterId)
   
        for i in range(numJoints):
            jointInfo = p.getJointInfo(self.baxterId, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                self.client.setJointMotorControl2(bodyIndex=self.baxterId, jointIndex=40, controlMode=p.POSITION_CONTROL,targetVelocity=0,
                                        targetPosition=jointPoses[qIndex-7])
    
    def accurateIK(self, targetPosition, targetOrientation,endEffectorId=48, 
               useNullSpace=False, maxIter=10, threshold=1e-4):
        closeEnough = False
        iter = 0
        dist2 = 1e30

        numJoints = self.client.getNumJoints(self.baxterId)

        while (not closeEnough and iter<maxIter):
            if useNullSpace:
                jointPoses = self.client.calculateInverseKinematics(self.baxterId, endEffectorId, targetPosition,targetOrientation=[0,1,0,0])
            else:
                jointPoses = self.client.calculateInverseKinematics(self.baxterId, endEffectorId, targetPosition)

            for i in range(numJoints):
                jointInfo = self.client.getJointInfo(self.baxterId, i)

                qIndex = jointInfo[3]
                if qIndex > -1:
                    self.client.resetJointState(self.baxterId,i,jointPoses[qIndex-7])
            ls = self.client.getLinkState(self.baxterId,endEffectorId)    
            newPos = ls[4]
            diff = [targetPosition[0]-newPos[0],targetPosition[1]-newPos[1],targetPosition[2]-newPos[2]]
            dist2 = np.sqrt((diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]))
#              print("dist2=",dist2)
            closeEnough = (dist2 < threshold)
            iter=iter+1
#        print("iter=",iter)
        return jointPoses
            
if __name__=='__main__':
    pass
        