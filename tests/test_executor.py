import math
from types import SimpleNamespace as Obj
import numpy as np
from closed_loop import Executor


def test_executor_holds_yaw_for_translation_then_rotates():
    events=[]
    class Future:
        def join(self): events.append("join")
    class Client:
        p=np.array([1.,2.,-3.]); yaw=math.pi/2
        def getMultirotorState(self):
            q=Obj(w_val=math.cos(self.yaw/2),z_val=math.sin(self.yaw/2),x_val=0.,y_val=0.)
            return Obj(kinematics_estimated=Obj(position=Obj(x_val=self.p[0],y_val=self.p[1],z_val=self.p[2]),orientation=q))
        def moveToPositionAsync(self,x,y,z,**kwargs):
            events.append(("move",kwargs)); self.p=np.array([x,y,z]); return Future()
        def rotateToYawAsync(self,angle,**kwargs):
            events.append(("rotate",angle,kwargs)); self.yaw=math.radians(angle); return Future()
        def hoverAsync(self): events.append("hover"); return Future()
        def simGetCollisionInfo(self): return Obj(has_collided=False)
    api=Obj(DrivetrainType=Obj(MaxDegreeOfFreedom=7),YawMode=lambda **k:Obj(**k))
    p,yaw=Executor(Client(),api).step([2.,0.,1.,-math.pi/2])
    assert np.allclose(p,[1.,4.,-4.]) and abs(yaw)<1e-9
    assert events[0][0]=="move" and events[1]=="join" and events[2][0]=="rotate"
    assert math.isclose(events[0][1]["yaw_mode"].yaw_or_rate,90) and events[0][1]["yaw_mode"].is_rate is False
    assert events[0][1]["drivetrain"]==7
    assert events[2][2]["margin"]==.01
