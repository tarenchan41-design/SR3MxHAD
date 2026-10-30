import math
from types import SimpleNamespace as Obj
import numpy as np
import pytest
from sr3m.sim import Executor


def test_executor_holds_yaw_for_translation_then_rotates():
    events=[]
    class Future:
        def join(self): events.append("join")
    class Client:
        p=np.array([1.,2.,-3.]); yaw=math.pi/2
        def getMultirotorState(self):
            q=Obj(w_val=math.cos(self.yaw/2),z_val=math.sin(self.yaw/2),x_val=0.,y_val=0.)
            return Obj(kinematics_estimated=Obj(position=Obj(x_val=self.p[0],y_val=self.p[1],z_val=self.p[2]),orientation=q,
                                                linear_velocity=Obj(x_val=0.,y_val=0.,z_val=0.)))
        def moveByVelocityAsync(self,vx,vy,vz,**kwargs):
            events.append(("move",kwargs)); self.p=self.p+np.array([vx,vy,vz]); return Future()
        def rotateToYawAsync(self,angle,**kwargs):
            events.append(("rotate",angle,kwargs)); self.yaw=math.radians(angle); return Future()
        def hoverAsync(self): events.append("hover"); return Future()
        def simGetCollisionInfo(self): return Obj(has_collided=False)
    api=Obj(DrivetrainType=Obj(MaxDegreeOfFreedom=7),YawMode=lambda **k:Obj(**k))
    p,yaw=Executor(Client(),api).step([2.,0.,1.,-math.pi/2])
    assert np.linalg.norm(p-[1.,4.,-4.])<=.2 and abs(yaw)<1e-9
    assert events[0][0]=="move" and "join" not in events[:events.index(next(e for e in events if e[0]=="rotate"))]
    rotate=next(e for e in events if e[0]=="rotate")
    assert math.isclose(events[0][1]["yaw_mode"].yaw_or_rate,90) and events[0][1]["yaw_mode"].is_rate is False
    assert events[0][1]["drivetrain"]==7
    assert rotate[2]["margin"]==.01


def test_executor_stops_at_target_before_rotating():
    class Client:
        p=np.zeros(3); yaw=0.; speed=1.; rotated_at=None; rotated_speed=None
        def getMultirotorState(self):
            q=Obj(w_val=math.cos(self.yaw/2),z_val=math.sin(self.yaw/2),x_val=0.,y_val=0.)
            return Obj(kinematics_estimated=Obj(position=Obj(x_val=self.p[0],y_val=self.p[1],z_val=self.p[2]),orientation=q,
                                                linear_velocity=Obj(x_val=self.speed,y_val=0.,z_val=0.)))
        def moveByVelocityAsync(self,vx,vy,vz,**kwargs):
            self.p=self.p+np.array([vx,vy,vz])*.5; self.speed*=.5; return Obj(join=lambda:None)
        def rotateToYawAsync(self,angle,**kwargs):
            self.rotated_at=self.p.copy(); self.rotated_speed=self.speed; self.yaw=math.radians(angle); return Obj(join=lambda:None)
        def hoverAsync(self): return Obj(join=lambda:None)
        def simGetCollisionInfo(self): return Obj(has_collided=False)
    client=Client()
    api=Obj(DrivetrainType=Obj(MaxDegreeOfFreedom=7),YawMode=lambda **k:Obj(**k))
    Executor(client,api).step([2.,0.,0.,.1])
    assert np.linalg.norm(client.rotated_at-[2.,0.,0.])<=.2 and client.rotated_speed<=.1


def test_executor_ignores_contact_before_it_starts_but_raises_on_collision():
    class Client:
        p=np.zeros(3); yaw=0.; collided=True
        def getMultirotorState(self):
            q=Obj(w_val=1.,z_val=0.,x_val=0.,y_val=0.)
            return Obj(kinematics_estimated=Obj(position=Obj(x_val=self.p[0],y_val=self.p[1],z_val=self.p[2]),orientation=q,
                                                linear_velocity=Obj(x_val=0.,y_val=0.,z_val=0.)))
        def moveByVelocityAsync(self,vx,vy,vz,**kwargs):
            self.p=self.p+np.array([vx,vy,vz]); return Obj(join=lambda:None)
        def rotateToYawAsync(self,angle,**kwargs): return Obj(join=lambda:None)
        def hoverAsync(self): return Obj(join=lambda:None)
        def simGetCollisionInfo(self):
            flag,self.collided=self.collided,False
            return Obj(has_collided=flag)
    client=Client()
    api=Obj(DrivetrainType=Obj(MaxDegreeOfFreedom=7),YawMode=lambda **k:Obj(**k))
    executor=Executor(client,api)
    executor.step([2.,0.,0.,0.])
    client.collided=True
    with pytest.raises(RuntimeError,match="Collision"): executor.step([2.,0.,0.,0.])
