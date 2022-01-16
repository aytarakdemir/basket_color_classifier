from time import sleep
import pybullet as p
import numpy as np
import cv2
import time
import pybullet_utils.bullet_client as bullet_client
position = []
color = ""


def detection():
    global green_pos
    global color
    
    frame = get_camera_image()
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
    
    # print("green: " + str(count_green))
    # print("red: " + str(count_red))
    # print("blue: " + str(count_blue))

    if count_green > 0:
        color = "Green"
    
    elif count_red > 0:
        color = "Red"
        
    elif count_blue > 0:
        color = "Blue"

    # print(color)
        
    # res = cv2.bitwise_and(frame, frame, mask= green_mask)
    # res2 = cv2.bitwise_and(frame, frame, mask= red_mask)
    # res3 = cv2.bitwise_and(frame, frame, mask= blue_mask)
    
    # cv2.imshow("frame", frame)
    # cv2.imshow("green", res)
    # cv2.imshow("red", res2)
    # cv2.imshow("blue", res3)
    # cv2.waitKey(0)    

def get_camera_image():
    view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0, 0, 0.78],
            cameraTargetPosition=[0, 0, .35],
            cameraUpVector=[0, 1, 0])

    projection_matrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=1.5)
            
    width, height, rgb, depth, seg = p.getCameraImage(
        width=480,
        height=320,
        projectionMatrix = projection_matrix,
        viewMatrix = view_matrix
        )
    return rgb

def setUpWorld(initialSimSteps=100):
    """
    Reset the simulation to the beginning and reload all models.
    Parameters
    ----------
    initialSimSteps : int
    Returns
    -------
    baxterId : int
    endEffectorId : int
    """
    p.resetSimulation()
    
    # Load plane
    p.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True)
    
    p.loadURDF("table/table.urdf", [-0.2, -0.2, -1], useFixedBase=True)
    
    p.loadURDF("basket/basket.urdf", [-0.7, 0, -1])
    p.loadURDF("basket/basket.urdf", [-0.7, -0.4, -1])
    p.loadURDF("basket/basket.urdf", [-0.4, -0.7, -1])

    sleep(0.1)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
    # Load Baxter
    baxterId = p.loadURDF("baxter_common/baxter_description/urdf/toms_baxter.urdf", useFixedBase=True)
    p.resetBasePositionAndOrientation(baxterId, [0.2, -0.8, 0.0], [0., 0., -1., -1.])
    # p.resetBasePositionAndOrientation(baxterId, [0.5, -0.8, 0.0],[0,0,0,1])
    #p.resetBasePositionAndOrientation(baxterId, [0, 0, 0], )

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

    # Grab relevant joint IDs
    endEffectorId = 48 # (left gripper left finger)

    # Set gravity
    p.setGravity(0., 0., -10)

    # Let the world run for a bit
    for _ in range(initialSimSteps):
        p.stepSimulation()

    return baxterId, endEffectorId
    
def getJointRanges(bodyId, includeFixed=False):
    """
    Parameters
    ----------
    bodyId : int
    includeFixed : bool

    Returns
    -------
    lowerLimits : [ float ] * numDofs
    upperLimits : [ float ] * numDofs
    jointRanges : [ float ] * numDofs
    restPoses : [ float ] * numDofs
    """

    lowerLimits, upperLimits, jointRanges, restPoses = [], [], [], []

    numJoints = p.getNumJoints(bodyId)

    for i in range(numJoints):
        jointInfo = p.getJointInfo(bodyId, i)

        if includeFixed or jointInfo[3] > -1:

            ll, ul = jointInfo[8:10]
            jr = ul - ll

            # For simplicity, assume resting state == initial state
            rp = p.getJointState(bodyId, i)[0]

            lowerLimits.append(-2)
            upperLimits.append(2)
            jointRanges.append(2)
            restPoses.append(rp)

    return lowerLimits, upperLimits, jointRanges, restPoses

def accurateIK(bodyId, endEffectorId, targetPosition, targetOrientation, 
               useNullSpace=False, maxIter=10, threshold=1e-4):
    """
    Parameters
    ----------
    bodyId : int
    endEffectorId : int
    targetPosition : [float, float, float]
    lowerLimits : [float] 
    upperLimits : [float] 
    jointRanges : [float] 
    restPoses : [float]
    useNullSpace : bool
    maxIter : int
    threshold : float

    Returns
    -------
    jointPoses : [float] * numDofs
    """
    closeEnough = False
    iter = 0
    dist2 = 1e30

    numJoints = p.getNumJoints(baxterId)

    while (not closeEnough and iter<maxIter):
        if useNullSpace:
            jointPoses = p.calculateInverseKinematics(bodyId, endEffectorId, targetPosition,targetOrientation=[0,1,0,0])
        else:
            jointPoses = p.calculateInverseKinematics(bodyId, endEffectorId, targetPosition)
    
        for i in range(numJoints):
            jointInfo = p.getJointInfo(bodyId, i)
            
            qIndex = jointInfo[3]
            if qIndex > -1:
                p.resetJointState(bodyId,i,jointPoses[qIndex-7])
        ls = p.getLinkState(bodyId,endEffectorId)    
        newPos = ls[4]
        diff = [targetPosition[0]-newPos[0],targetPosition[1]-newPos[1],targetPosition[2]-newPos[2]]
        dist2 = np.sqrt((diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]))
#        print("dist2=",dist2)
        closeEnough = (dist2 < threshold)
        iter=iter+1
#    print("iter=",iter)
    return jointPoses

def setMotors(bodyId, jointPoses):
    """
    Parameters
    ----------
    bodyId : int
    jointPoses : [float] * numDofs
    """
    
    numJoints = p.getNumJoints(bodyId)
   
    for i in range(numJoints):
        jointInfo = p.getJointInfo(bodyId, i)
        qIndex = jointInfo[3]
        if qIndex > -1:
            p.setJointMotorControl2(bodyIndex=bodyId, jointIndex=40, controlMode=p.POSITION_CONTROL,targetVelocity=0,
                                    targetPosition=jointPoses[qIndex-7])
   
def randomSpawn():
    global position
    global cubeOrn

    arr = np.random.randint(3,size=1)
    rnd = np.random.randint(4,size=1)
    arr1 = np.random.randint(300,size=2)
    arr1+=50
    x = -arr1[0]/1000
    y = -arr1[1]/1000
    
    
    global box_id
    # box_id = p.loadURDF("cube_green.urdf", [x, y, -0.35],globalScaling=0.42)
    if arr[0]==0:
        box_id = p.loadURDF("cube_blue.urdf", [x, y, -0.35],globalScaling=0.42)
        pass
    elif arr[0]==1:
        pass
        box_id = p.loadURDF("cube_red.urdf", [x, y, -0.35],globalScaling=0.42)
    elif arr[0]==2:
        box_id = p.loadURDF("cube_green.urdf", [x, y, -0.35],globalScaling=0.42)

    position, cubeOrn = p.getBasePositionAndOrientation(box_id)
    

def moveTo(targetPosition, prevPos = [0,0,0], gripperClosed = False):
    currentPos = prevPos.copy()
    for i in range(200):
        currentPos[0] = currentPos[0] + ((targetPosition[0] - prevPos[0]) / 200)
        currentPos[1] = currentPos[1] + ((targetPosition[1] - prevPos[1]) / 200)
        currentPos[2] = currentPos[2] + ((targetPosition[2] - prevPos[2]) / 200)
        # lowerLimits, upperLimits, jointRanges, restPoses = getJointRanges(baxterId, includeFixed=False)
        jointPoses = accurateIK(baxterId, endEffectorId, currentPos, [0,0,1,0], useNullSpace=useNullSpace)
        setMotors(baxterId, jointPoses)
        if gripperClosed:
            position, cubeOrn = p.getBasePositionAndOrientation(box_id)
            position = list(position)
            # Updating the position of the cube as if it was the gripper, so it stays fixed to the gripper.
            position[0] = position[0] + ((targetPosition[0] - prevPos[0]) / 200)
            position[1] = position[1] + ((targetPosition[1] - prevPos[1]) / 200)
            position[2] = position[2] + ((targetPosition[2] - prevPos[2]) / 200)
            p.resetBasePositionAndOrientation(box_id, position, cubeOrn)
        runNSteps(1)

def openGripper():
    p.setJointMotorControl2(bodyIndex = baxterId, jointIndex = 49, controlMode = p.POSITION_CONTROL, force = 100, targetPosition = 1)
    p.setJointMotorControl2(bodyIndex = baxterId, jointIndex = 51, controlMode = p.POSITION_CONTROL, force = 100, targetPosition = -1)
    runNSteps(100)

def closeGripper():
    p.setJointMotorControl2(bodyIndex = baxterId, jointIndex = 49, controlMode = p.POSITION_CONTROL, force = 100, targetPosition = -1)
    p.setJointMotorControl2(bodyIndex = baxterId, jointIndex = 51, controlMode = p.POSITION_CONTROL, force = 100, targetPosition = 1)
    runNSteps(100)
  
def runNSteps(n):
    for i in range(n):
        p.stepSimulation()
    	
    
if __name__ == "__main__":
    
    guiClient = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(2., 180, 0., [0.52, 0.2, np.pi/4.])
    nullSpaceId = p.addUserDebugParameter("nullSpace",0,1,0)
    baxterId, endEffectorId = setUpWorld()
    lowerLimits, upperLimits, jointRanges, restPoses = getJointRanges(baxterId, includeFixed=False)
    maxIters = 100000
    success=0
    total=0
    time.sleep(2.)

    p.getCameraImage(320,200, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    
    previousLoopPos= [0, 0, 0]
    
    for i in range(maxIters):
        p.stepSimulation()
        nullSpace = p.readUserDebugParameter(nullSpaceId)
        nullSpaceId = p.addUserDebugParameter("nullSpace",0,1,0)
        useNullSpace = nullSpace > 0.5

        randomSpawn()
        detection()
        
        runNSteps(200)
        
        gripperClosed = False

        # Position the hand just above the cube 
        targetPosition = [position[0],position[1],-0.30]
        moveTo(targetPosition, previousLoopPos, gripperClosed)
        
        # Open the gripper
        gripperClosed = False
        openGripper()
        
        # Lower the gripper
        prevPos = targetPosition.copy()
        targetPosition = [position[0],position[1],-0.35]
        moveTo(targetPosition, prevPos, gripperClosed)
        
        # Close the gripper
        closeGripper()
        gripperClosed = True

        # Lift the hand
        prevPos = targetPosition.copy()
        targetPosition = [position[0],position[1],-0.20]
        moveTo(targetPosition, prevPos, gripperClosed)
        
        # Move to the corresponding basket
        prevPos = targetPosition.copy()
        if color == "Red":
            targetPosition = [-0.7, 0,-0.20]
            moveTo(targetPosition, prevPos, gripperClosed)
        elif color == "Blue":
            targetPosition = [-0.7, -0.4,-0.20]
            moveTo(targetPosition, prevPos, gripperClosed)
        elif color == "Green":
            targetPosition = [-0.4, -0.7,-0.20]
            moveTo(targetPosition, prevPos, gripperClosed)
        else:
            # If color undefined, throw it on the ground 
            targetPosition = [-0.7, -0.7,-0.20]
            moveTo(targetPosition, prevPos, gripperClosed)


        previousLoopPos = targetPosition.copy()
        
        # Open the gripper
        openGripper()
        gripperClosed = False
        pos=p.getBasePositionAndOrientation(box_id)[0]
        # success+=1
        if color == "Red":
            # print(pos)
            targetPosition = [-0.7, 0,-0.20]
            if -0.9<pos[0]<-0.5 and -0.2 <pos[1]< 0.2:
                success+=1
        elif color == "Blue":
            # print(pos)
            targetPosition = [-0.7, -0.4,-0.20]
            if -0.9<pos[0]<-0.5 and -0.6 <pos[1]< -0.2:
                success+=1
        elif color == "Green":
            # print(pos)
            targetPosition = [-0.4, -0.7,-0.20]
            if -0.6<pos[0]<-0.2 and -0.9 <pos[1]< -0.5:
                success+=1
        total+=1
        # runNSteps(10)
        p.removeBody(box_id)
        print("Successful: ",success)
        print("Failed: ",(total-success))
        print("Success rate: ",str(success*100/total))

